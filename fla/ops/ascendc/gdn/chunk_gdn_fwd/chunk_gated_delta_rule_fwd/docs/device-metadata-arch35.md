# npu_chunk_gated_delta_rule_fwd：`cu_seqlens` / `chunk_indices` 改为 device 参数（arch35 路径分析）

> 范围：Ascend950（`NpuArch::DAV_3510`，arch35）路径，即
> `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd` 融合算子的
> `arch35` tiling + `arch35` kernel。arch22（A2/A3）代码只在与本次 ABI 共享的位置
> 提示需要同步修改，不做方案设计。
>
> 状态：方案分析（§0–§6）+ 已完成的实现记录（§7）。实现只覆盖 arch35 目标路径与
> 其共享的 host/ABI 代码，未执行 NPU 上的编译与精度验证。

## 0. 结论

1. **kernel 侧已经完全是 device 语义**：arch35 kernel 的入口参数就是
   `GM_ADDR cu_seqlens, GM_ADDR chunk_indices`
   (`op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp:477`)，
   `BuildHoPipelineContext`、KKT cube、local cumsum、solve_tri、recompute、FwdH、FwdO、
   `WritePublicCumsumRows` 全部用 `GlobalTensor<int64_t>` + `SetGlobalBuffer` + `GetValue`
   读 GM。**kernel 不需要改**。
2. **host 侧只有 2 处真正的“读值”阻塞点**（其余都是读 shape）：
   - `op_host/op_tiling/arch35/chunk_gated_delta_rule_fwd_arch35_state_output_tiling.cpp:251-267`
     → `GetOptionalInputTensor(...)->GetData<int64_t>()`；
   - 其下游 `.../operators/recompute_w_u_fwd/op_host/op_tiling/recompute_w_u_fwd_tiling_processor.h:374-440`
     （`VariableLenTiling` 对 cu/chunk_indices 的值做校验）。
   把这两处去掉或改成“有 host 指针才校验”即可让 tiling 不再依赖 host 值。
3. **ABI 需要改 4 层**：op_def 的 `ValueDepend` 去掉；L0 `l0op::ChunkGatedDeltaRuleFwd`
   的 `aclIntArray*` → `aclTensor*`；L2 `aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize`
   同上；host 适配（stable cpp / `_stable.py` / `_aclnn_ctypes.py`）改为传 device 张量、
   用 shape 推导输出 shape。
4. **级联依赖**：该 op 的 L2 是复合入口，A5 默认形状会走
   `ChunkGatedDeltaRuleFwdPrepare` + `ChunkFwdH` + `ChunkFwdO`（见
   `aclnn_chunk_gated_delta_rule_fwd.cpp:684` 的 `UsePreparePath`）。如果要把公开
   ABI 换成 device 张量，这三个兄弟算子的 metadata ABI 必须同步迁移（它们的 tiling
   只读 shape，迁移是机械工作），否则 L2 无法把 device 张量再变回 `aclIntArray`。
   若只想让 arch35 **融合**路径先落地，则新增一个只走融合分支的 L2 符号（方案 B）。

## 1. 现状：值被谁读

| # | 位置 | 行为 | 阻塞 device 化 |
| --- | --- | --- | --- |
| 1 | `torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py:1524` | `_host_ints(cu_seqlens)` 造 **CPU** int64 张量（含 `_INT_CACHE`） | 是（host 侧入口） |
| 2 | `torch_custom/fla_npu/csrc/src/stable_chunk_gated_delta_rule_fwd.cpp:86-87` | `int_values()` 直接解引用 CPU 张量内存出 `std::vector` | 是 |
| 3 | 同上 `:115`、`:133-134` | 用值算 `final_state` 的 `seqNum`、`h` 的 chunk 数 | 是（可改 shape） |
| 4 | `_aclnn_ctypes.py:3061-3062` | ctypes 路径 `ctx.int_array(...)` 建 `aclIntArray` | 是（第二调用方） |
| 5 | L2 `aclnn_chunk_gated_delta_rule_fwd.cpp:226-246, 261-293, 532-536` | `SeqNum`/`ExpectedChunks`/`CheckMetadata` 读值做校验和 shape | 是 |
| 6 | L0 `op_api/chunk_gated_delta_rule_fwd.cpp:19-32, 63-69` | `ConvertToTensor(aclIntArray)` 造 host 常量张量 | 是（多一次 H2D） |
| 7 | `chunk_gated_delta_rule_fwd_arch35_state_output_tiling.cpp:251-267` | `GetData<int64_t>()` 取 host 指针 | **是（核心阻塞）** |
| 8 | `recompute_w_u_fwd_tiling_processor.h:374-440` | 值校验：`cu[0]==0`、单调、`chunk_indices` canonical、两侧长度一致 | **是（核心阻塞）** |
| 9 | arch35 tiling 其余部分 | 只读 shape：`GetChunkCount(chunkShape)`（`arch35_tiling.cpp:82-97,208-219`）、`IsShape(cuShape,{2})`（:237）、`cuSeqlensDim0 = dim0-1`（state_output:291） | 否 |
| 10 | arch35 kernel | 全部 GM 读 | 否 |

补充：`def.cpp:40-43` 的 `ValueDepend(OPTIONAL)` 是“框架保证 host 上有数据”的声明，
有了它 tiling 才敢 `GetData`。去掉它是 device 化的前提，也意味着**任何 tiling 代码都不
能再解引用这两个张量**。

## 2. 目标契约

- `cu_seqlens`：可选，`[N+1]`，`INT64`，ND，device 张量；首项 0、单调不降、末项 == T。
- `chunk_indices`：可选，`[2*Nc]`（rank-1），`INT64`，ND，device 张量；canonical
  sequence-major 顺序。
- 二者同时存在或同时缺省；变长时物理 `B=1`。
- **shape 必须静态已知**（`N`、`Nc` 在 tiling 时可见），只有“值”是自由的。原因：
  `NT`（chunk 数）来自 `chunk_indices` 的 shape，直接进入 `taskNum`、workspace 布局、
  tiling 结构体和 kernel 启动参数；没有 `NT` 就无法 tiling（见 3.4 的失败分支）。
- host 不再做值校验（见 5.2 的处理方式）。

## 3. 修改清单

### 3.1 op_def（`op_host/chunk_gated_delta_rule_fwd_def.cpp:40-43`）

```cpp
this->Input("cu_seqlens").ParamType(OPTIONAL).DataType(indexTypes)   // 去掉 .ValueDepend(OPTIONAL)
    .Format(formats).UnknownShapeFormat(formats).AutoContiguous();
this->Input("chunk_indices").ParamType(OPTIONAL).DataType(indexTypes)
    .Format(formats).UnknownShapeFormat(formats).AutoContiguous();
```

- 保留 `indexTypes = {DT_INT64,...}`（kernel 硬编码 `int64_t`，不支持 int32）。
- `AutoContiguous()` 可保留（1-D 元数据基本恒连续；非连续时插 `Contiguous` 只读 shape）。
- 同时要改的兄弟 def（方案 A）：`chunk_gated_delta_rule_fwd_prepare_def.cpp:46`、
  `chunk_fwd_h_def.cpp:64-74`、`chunk_fwd_o_def.cpp` 对应输入。
- arch22 用的是同一份 def，因此 **A2/A3 也一并变成 device 语义**，arch22 tiling 的两处
  读值（`.../arch22/chunk_gated_delta_rule_fwd_arch22_state_output_tiling.cpp:275-279`
  和 `.../internal/arch22/.../recompute_w_u_fwd_tiling_processor.h:365-405`）必须同步
  按 3.4 处理，否则 A2/A3 会解引用 device 指针。

### 3.2 L0（`op_host/op_api/chunk_gated_delta_rule_fwd.{h,cpp}`）

```cpp
// .h
const aclTensor *cuSeqlensOptional,
const aclTensor *chunkIndicesOptional,
```

```cpp
// .cpp：删除 ConvertMetadata 与 nullptr 配对检查，直接下传
const aclError ret = ADD_TO_LAUNCHER_LIST_AICORE(
    ChunkGatedDeltaRuleFwd,
    OP_INPUT(q, k, v, beta, aStorage, rawG, gkOptional, initialStateOptional,
             cuSeqlensOptional, chunkIndicesOptional),   // 原为 ConvertMetadata(...) 结果
    ...
```

`ConvertMetadata` 里 `executor->ConvertToTensor(aclIntArray, DT_INT64)` 是“host 数组 →
常量张量”的转换；删除后 ABI 里是普通 device 张量，框架不再插入额外拷贝。

### 3.3 L2（`op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.{h,cpp}`）

签名：`const aclIntArray *` → `const aclTensor *`（`aclnn_..._fwd.h:25-26`、
`..._fwd.cpp:607-608, 891-892`；`ChunkGatedDeltaRuleFwdParams:54-55` 同步）。

checker 全部改成 shape-only：

```cpp
static int64_t SeqNum(const ChunkGatedDeltaRuleFwdParams &params, int64_t batch)
{
    return params.cuSeqlensOptional == nullptr
               ? batch
               : Dim(params.cuSeqlensOptional, 0) - 1;      // 原 aclIntArray->Size()-1
}

static int64_t ChunkCount(const ChunkGatedDeltaRuleFwdParams &params, int64_t seqlen)
{
    if (params.chunkIndicesOptional == nullptr) {
        return (seqlen + params.chunkSize - 1) / params.chunkSize;
    }
    return Dim(params.chunkIndicesOptional, 0) / 2;          // 原对 cu 值求和
}

static aclnnStatus CheckMetadata(const ChunkGatedDeltaRuleFwdParams &params)
{
    if (params.cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_COND(Rank(params.cuSeqlensOptional) == 1 &&
                   Dim(params.cuSeqlensOptional, 0) >= 2, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens must be rank-1 with at least two elements.");
    CHECK_COND(params.cuSeqlensOptional->GetDataType() == DataType::DT_INT64 &&
                   params.chunkIndicesOptional->GetDataType() == DataType::DT_INT64,
               ACLNN_ERR_PARAM_INVALID, "cu_seqlens/chunk_indices must be int64.");
    CHECK_COND(Rank(params.chunkIndicesOptional) == 1 &&
                   Dim(params.chunkIndicesOptional, 0) > 0 &&
                   Dim(params.chunkIndicesOptional, 0) % 2 == 0,
               ACLNN_ERR_PARAM_INVALID,
               "chunk_indices must be a rank-1 (seq,chunk) pair list.");
    // 值语义（首项 0 / 单调 / 末项 T / canonical 顺序）不再能在 host 校验，见 5.2。
    return ACLNN_SUCCESS;
}
```

其它用值的地方：

- `CheckParams`：`chunks = ExpectedChunks(...)` → `ChunkCount(params, info.seqlen)`（:479），
  用于 `hOutOptional` 的 `[B,NT,Hv,K,V]` 校验（:521-525）；
- `SeqNum` 调用于 :538、:662（`finalStateOutOptional` / `initialStateOptional` 的
  `[N,Hv,K,V]` 校验、prepare 分支 `stateShape`），改完后语义不变；
- prepare 分支的 `hShape`（:693）用 `ChunkCount`；
- 融合分支（:861-867）把 `params.cuSeqlensOptional/chunkIndicesOptional` 直接传给 L0。
- 刻意保留的检查：`(cu == nullptr) == (chunk == nullptr)`、`cu→B==1`、`chunkSize∈{64,128}`、
  rank/dtype/长度，这些只依赖 descriptor。

### 3.4 tiling（arch35）

**a) `arch35_tiling.cpp` 不需要改值逻辑**（已确认只读 shape）。建议补两点健壮性：

```cpp
// 静态 shape 要求：NT 来自 chunk_indices 的 dim0
OP_CHECK_IF(isVarlen && (cuShape->GetStorageShape().GetDim(0) < 0 ||
                         chunkShape->GetStorageShape().GetDim(0) < 0),
            OP_LOGE(context->GetNodeName(),
                    "Device cu_seqlens/chunk_indices require static lengths "
                    "(sequence count and chunk count size the workspace)."),
            return ge::GRAPH_FAILED);
```

**b) `chunk_gated_delta_rule_fwd_arch35_state_output_tiling.cpp:251-267`：删掉取指针**

```cpp
-    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS);
-    auto chunkIndicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES);
-    const int64_t *cuSeqlensData = cuSeqlensTensor == nullptr ? nullptr : cuSeqlensTensor->GetData<int64_t>();
-    const int64_t *chunkIndicesData = chunkIndicesTensor == nullptr ? nullptr : chunkIndicesTensor->GetData<int64_t>();
+    // Device metadata: the values are consumed by the kernel only. Host tiling must
+    // never dereference them (no ValueDepend, so GetData would be a device address).
     RecomputeWUFwdTilingContext recomputeContext{
         context->GetNodeName(), ..., 
         context->GetOptionalInputShape(INPUT_CU_SEQLENS),
         context->GetOptionalInputShape(INPUT_CHUNK_INDICES),
-        cuSeqlensData,
-        chunkIndicesData,
+        /*cuSeqlensData=*/nullptr,
+        /*chunkIndicesData=*/nullptr,
         ...};
+    recomputeContext.metadataOnDevice = isVarlen;
```

`hContext.cuSeqlensDim0 = tokenBatch + 1`（:291）已经来自 shape，保持不变；
`tokenBatch`（H tiling 用）本来就是 `cuShape.dim0 - 1`，也保持不变。

**c) `recompute_w_u_fwd_tiling_processor.h`（arch35:374-440，arch22 同构）：值校验降级**

给 `RecomputeWUFwdTilingContext` 加一个字段（默认 false，保持独立
`RecomputeWUFwd` 算子的旧行为）：

```cpp
bool metadataOnDevice = false;   // device 张量：只校验 shape，值由 kernel 消费
```

```cpp
ge::graphStatus VariableLenTiling()
{
    ... // shape 校验（cu rank-1 且 dim0>=2、chunk rank-1）全部保留
    const int64_t seqlensDim0 = ctx_.cuSeqlensShape->GetStorageShape().GetDim(0);
    const int64_t chunkIndicesDim0 = ctx_.chunkIndicesShape->GetStorageShape().GetDim(0);
    OP_CHECK_IF(chunkIndicesDim0 <= 0 || chunkIndicesDim0 % 2 != 0, ...);

    if (!ctx_.metadataOnDevice) {          // 仅旧 host 数组模式做值校验
        ... // 原有 cu[0]==0 / 单调 / expectChunkIndices / 长度一致校验
    }
    tiling_.chunkNum = chunkIndicesDim0 / 2;
    return ge::GRAPH_SUCCESS;
}
```

注意 `Process()` 里对 varlen 的 `tiling_.B == 1` 检查（:466-471）只读 shape，保留。

**d) 其它 arch35 tiling**：`arch35_tiling.cpp` 的 `GetChunkCount` 支持 1-D（dim0 偶数）
与 2-D `[Nc,2]`，而 recompute 处理器要求 rank-1。现状就存在这一不一致，device 化时建议
统一成 **rank-1 `[2*Nc]`**（README 与 ABI 注释都写的是 `[2*Nc]`），或两处同时支持 2-D。

### 3.5 host 适配

**a) `torch_custom/fla_npu/csrc/src/stable_chunk_gated_delta_rule_fwd.cpp`**

```cpp
-  const std::vector<int64_t> cu = int_values(cu_seqlens);
-  const std::vector<int64_t> ci = int_values(chunk_indices);
...
   out_final_state = allocate_sizes(
-      {layout_math::sequences(cu, batch), heads, state_tail_k, state_tail_v},
+      {layout_math::sequences_of(cu_seqlens, batch), heads, state_tail_k, state_tail_v},
       state_dtype, q_meta);
...
   const std::vector<int64_t> h_shape = {
-      batch, layout_math::chunks(cu, ci, chunk_size, tokens), heads,
+      batch, layout_math::chunks_of(chunk_indices, chunk_size, tokens), heads,
       state_tail_k, state_tail_v};
...
   FLA_STABLE_EXEC("aclnnChunkGatedDeltaRuleFwd", ...,
-      optional_tensor(initial_state), int_array(cu), int_array(ci),
+      optional_tensor(initial_state), optional_tensor(cu_seqlens),
+      optional_tensor(chunk_indices),
       cstr(...), ...);
```

`layout_math.h` 新增 shape 版本（保留原值版本给仍用 `aclIntArray` 的适配器）：

```cpp
// 变长时 cu_seqlens 的 numel-1 就是序列数；dense 时是 batch。
inline int64_t sequences_of(const std::optional<torch::stable::Tensor>& cu,
                            int64_t batch_size);

// chunk_indices 优先（精确、且允许空段）；否则按 T 上取整。
inline int64_t chunks_of(const std::optional<torch::stable::Tensor>& chunk_indices,
                         int64_t chunk_size, int64_t tokens);
```

`int_values` 会按 `meta.data` 解引用；对 device 张量这是非法访问，必须彻底删掉这两次调用。

**b) `torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py`**

```python
-        _host_ints(cu_seqlens), _host_ints(chunk_indices),
+        _device_ints(cu_seqlens), _device_ints(chunk_indices),
```

`_device_ints`：`None` → `None`；已是 NPU 张量直接透传（不 clone、不改 dtype 语义）；
list/tuple → `torch.tensor(..., dtype=torch.int64, device=<q.device>)`。可选优化：
list 输入时校验放在这里（现在只有 ctypes 路径有值校验），device 张量输入时信任调用方。

**c) `_aclnn_ctypes.py`**

- `_GET_WORKSPACE_ARGTYPES["aclnnChunkGatedDeltaRuleFwd"]` 两个 `c_void_p` 不变（
  `aclIntArray*` 与 `aclTensor*` 都是 `void*`），`aclnn_abi_contract.py` 的参数名/类型
  合同也不需要改；
- `ctx.int_array(cu_seqlens)` → `ctx.tensor(<device int64 张量>, "cu_seqlens")`；
- 因为这条路径的 Python 入参还是 list，保留现有值校验（:2975-2989）与
  `seq_num/chunks` 的 Python 侧推导（:3023、:3032-3039），再构造 device 张量下传。

### 3.6 方案 A 的级联（prepare / ChunkFwdH / ChunkFwdO）

`UsePreparePath()`（`aclnn_chunk_gated_delta_rule_fwd.cpp:103-133`）在 A5 上按形状选择
prepare 分支（BF16、K=V=128、chunk=64、Hv/Hk≤4），这是默认模型 case 走的分支。
公开 ABI 换成 device 张量后，L2 必须把同一个 device 张量继续传给这三个子算子：

- `chunk_gated_delta_rule_fwd_prepare`：`aclnn_..._prepare.{h,cpp}`、
  `chunk_gated_delta_rule_fwd_prepare.cpp:45,90,114`（删除 `ConvertIntArrayToTensor`）、
  `chunk_gated_delta_rule_fwd_prepare_def.cpp:46`（去 `ValueDepend`）；
- `chunk_fwd_h`：`aclnn_chunk_fwd_h.h:50-51`、`chunk_fwd_h_def.cpp:64-74`
  （tiling `chunk_fwd_h_tiling.cpp:88` 只用到 tensor 存在性/dtype，无值读）；
- `chunk_fwd_o`：同理。

已确认这三个算子的 tiling 都不读值（仅 shape/dtype），因此迁移是机械的；但**必须在同一
次改动内完成**，否则 A5 默认 case 会在 L2 里无处安放 device 元数据。

如果要让 arch35 融合路径先行、不动兄弟算子：采用**方案 B**——新增
`aclnnChunkGatedDeltaRuleFwdV2`（device 版），内部只保留融合分支；prepare 分支在 V2 里
直接拒绝并给出明确错误码，待兄弟算子迁移后再打开。代价是 V2 在 A5 默认形状上不可用。

> 实际落地时采用了介于 A、B 之间的做法：只给三个兄弟算子的 **L0** 增加 `aclTensor*` 重载
> （公开 aclnn ABI、`op_def`、以及 KDA/GDN-bwd 复合调用方全部保持不变），详见 §7。


## 4. 风险与边界

1. **静态 shape 约束**：`N`、`Nc` 必须可在 tiling 期得到。若上游给动态维
   （`dim0 = -1`），当前 tiling 会因 `GetChunkCount` 失败报 161002；建议按 3.4(a) 给出
   可读错误，并把“元数据 shape 静态”写进 `docs/api.md`/README。
2. **丢失 host 值校验**（`cu[0]==0`、单调、`cu[-1]==T`、canonical 顺序、两侧长度一致）。
   这些一旦违规，kernel 会按非法 `seq_id`/`bos`/`eos` 读 `cu` 或写 `gCumsumBth`/`h`
   workspace，属于 OOB。三种处置，建议 3 + 1：
   - 继续在 Python 入口对 list 输入校验（已有，保留）；
   - device 输入按契约信任（`recurrent_kda` 的 `cu_seqlens` 也是这个策略，见其
     `docs/api.md` 的“offset 必须单调不减”约定）；
   - 可选：在 `RunPhase6` 的 AIV block 0 加一次轻量 device 校验（`cu[0]==0`、
     `cu[last]==tokens`、`chunk_count(cu) == NT`），把结果写进 workspace 的一个 flag，
     所有 block 读完 flag 后 `return`/报错。仅在 debug 或 `hasCuSeqlens` 时开启，避免
     常规路径多一次全 block 同步。
3. **arch22 同步**：def/L0/L2 共享，A2/A3 必须同时去掉 `GetData`（`arch22_state_output_tiling.cpp:275-279`、
   `internal/arch22/.../recompute_w_u_fwd_tiling_processor.h:365-405`），否则 A2/A3 上
   `GetData` 返回的是 device 地址，会直接崩 tiling。
4. **ABI 兼容性**：`aclnnChunkGatedDeltaRuleFwd` 是 `extern "C"` 符号，参数类型从
   `aclIntArray*` 变 `aclTensor*` 后，仓外旧调用方（拿着 aclIntArray 的）会静默错调。
   仓内两个调用方（stable 适配、ctypes）需同步改；若要保留旧 ABI，只能走方案 B 的新符号。
5. **dtype**：只支持 `INT64`（kernel `int64_t`）。`recurrent_kda` 支持 int32 是因为它按
   tiling 里的 `cuSeqlensDtype` 分支读；本算子若要 int32，需要给 arch35 kernel 增加同类
   分支，不建议在本次改动里做。
6. **收益**：去掉 `_host_ints` 的 CPU 张量构造（约 18us/新长度）、`ConvertToTensor` 的
   常量张量与 H2D 拷贝、`ValueDepend` 带来的 host 数据准备；元数据可由上游 kernel 在
   device 上生成，便于图捕获/避免 host 同步。

## 5. 建议实施顺序

1. **Step 1（无 ABI 变化）**：tiling 去值化。传 `nullptr` + `metadataOnDevice`，
   `ValueDepend` 暂时保留（L2 仍传 `aclIntArray`，经 `ConvertToTensor` 变成 device 张量，
   tiling 不再读值）。跑 ATK 全量（含 varlen / B30 / V=256 case）确认 tiling 数值与
   workspace 完全一致（`NT`、`taskNum`、`h/vNew/ready` 偏移）。
2. **Step 2（ABI 变化）**：去 `ValueDepend`，L0/L2 换 `aclTensor*`，改 stable 适配 +
   `_stable.py` + `_aclnn_ctypes.py`，按方案 A 迁移 prepare/H/O（或先落方案 B 的 V2 入口）。
3. **Step 3**：清理 `int_array`/`int_values` 残留（若其它算子仍用则保留 helper），
   补 arch22 同构修改，跑完整回归。

## 6. 验证清单

- `python3 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py`（参数名/顺序不变，
  ctypes 类型仍是 `c_void_p`，应继续通过）；
- `python3 torch_custom/fla_npu/tools/op_abi_parity.py`（schema 已是 `Tensor?`，
  adapter 仍是 `std::optional<Tensor>`）与 `op_abi_validate.py`（aclnn 头 vs 转发实参）；
- `torch_custom/fla_npu/test/test_aclnn_ctypes_abi.py`（captured args 里 metadata 由
  int_array 变 tensor 的断言）；
- ATK 精度/性能：`tests/atk/chunk_gated_delta_rule_fwd`（dense + varlen、chunk 64/128、
  V=128/256、B30 case、`output_final_state`/`h` 导出）；
- 覆盖“cu/chunk_indices 为 device 张量”的新用例：值非法（未排序、末项≠T）时确认是
  报错或按契约未定义，不会静默写坏显存；
- 图模式（如用 ATC/OM）：确认 `ValueDepend` 去掉后没有其它 host 值依赖。

## 7. 实施记录（已完成）

按“先 tiling 去值、再改 ABI”的顺序落地，改动文件：

**tiling（arch35 + arch22 共享 ABI，必须一起改）**
- `op_kernel/internal/operators/recompute_w_u_fwd/op_host/op_tiling/recompute_w_u_fwd_tiling_processor.h`：
  context 新增 `metadataOnDevice`，`VariableLenTiling` 的 cu/chunk_indices **值校验**放在
  `if (!ctx_.metadataOnDevice)` 内；shape 校验（rank、`dim0>=2`、pair 数偶）与
  `chunkNum = dim0/2` 始终执行。
- `op_kernel/internal/arch22/.../recompute_w_u_fwd_tiling_processor.h`：同构修改（A2/A3 共享 def/L0/L2）。
- `op_host/op_tiling/arch35/chunk_gated_delta_rule_fwd_arch35_state_output_tiling.cpp`：
  删除 `GetOptionalInputTensor()->GetData<int64_t>()`，传 `nullptr` 并置 `metadataOnDevice = isVarlen`。
- `op_host/op_tiling/arch22/chunk_gated_delta_rule_fwd_arch22_state_output_tiling.cpp`：同上。
- `op_host/op_tiling/arch35/chunk_gated_delta_rule_fwd_arch35_tiling.cpp`：新增“device 元数据要求
  静态长度”的显式检查（`N`/`Nc` 决定 workspace）。

**融合算子 ABI**
- `op_host/chunk_gated_delta_rule_fwd_def.cpp`：`cu_seqlens`/`chunk_indices` 去掉 `ValueDepend`。
- `op_host/op_api/chunk_gated_delta_rule_fwd.{h,cpp}`：`aclIntArray*` → `aclTensor*`，删除
  `ConvertMetadata` 与 null 配对检查。
- `op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.{h,cpp}`：签名改 `aclTensor*`；
  `SeqNum`/`ChunkCount` 改 shape 推导；`CheckMetadata` 降级为 rank/dtype/长度检查；
  `ExpectedChunks` 全部替换为 `ChunkCount`。

**复合子算子（prepare / ChunkFwdH / ChunkFwdO）**
- `chunk_gated_delta_rule_fwd_prepare/op_host/op_api/chunk_gated_delta_rule_fwd_prepare.{h,cpp}`
- `chunk_fwd_h/op_host/op_api/chunk_fwd_h.{h,cpp}`
- `chunk_fwd_o/op_host/op_api/chunk_fwd_o.{h,cpp}`

  三个 L0 各保留**两个重载**：`aclTensor*`（device，复合路径用）与 `aclIntArray*`（host，
  历史 aclnn 入口用；内部仍 `ConvertToTensor` 后转调 device 版）。这样
  `aclnnChunkFwdH`/`aclnnChunkFwdO`/`aclnnChunkGatedDeltaRuleFwdPrepare` 的公开 C ABI、
  它们的 `op_def`（仍 `ValueDepend`）、以及 KDA fwd/bwd、GDN bwd 等复合调用方都不需要改。

**host 适配**
- `torch_custom/fla_npu/csrc/include/stable/layout_math.h`：新增 `sequences_of` / `chunks_of`（按 shape）。
- `torch_custom/fla_npu/csrc/src/stable_chunk_gated_delta_rule_fwd.cpp`：
  删除 `int_values`，改传 `optional_tensor(cu_seqlens/chunk_indices)`，输出 shape 用 `*_of`。
- `torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py`：新增 `_device_ints`（list → 目标
  device 上的 int64 张量，已有张量归一化后透传），`npu_chunk_gated_delta_rule_fwd` 改用它。
- `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py`：metadata 由 `ctx.int_array`
  改为 `ctx.int_tensor(..., q.device)`；Python 侧仍保留对 list 的值校验与输出 shape 推导。
- `torch_custom/fla_npu/test/test_aclnn_ctypes_abi.py`：`FakeCallContext` 增加 `int_tensor`。

**未改动（刻意）**
- arch35 kernel：本就全 device（GM）读取。
- prepare/ChunkFwdH/ChunkFwdO 的 `op_def` 与公开 aclnn ABI、KDA/GDN-bwd 复合入口。
- `chunk_fwd_h`/`chunk_fwd_o`/`prepare` 的 device 输入下游若仍声明 `ValueDepend`，其 tiling
  不读值，仅可能多一次元数据 D2H；如后续去掉它们的 `ValueDepend`，需要同时确认这三条
  历史 host 数组调用链的 H2D 拷贝仍由 opdev 的 host tensor 标记触发（属于后续独立验证项）。

**待验证（无 NPU 环境，未执行）**
- 编译：`op_host`/`op_api`/`op_kernel` 全量构建；
- ATK：`tests/atk/chunk_gated_delta_rule_fwd` dense/varlen、chunk 64/128、V=128/256、b30 case；
- `python3 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py`（参数名/数量与 ctypes
  类型未变，应通过）；
- `python tools/op_abi_parity.py`、`python tools/op_abi_validate.py`；
- A2/A3 回归（arch22 tiling 已同步去值化）。

