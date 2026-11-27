/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "chunk_fwd_o.h"
#include <string>

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkFwdO);

namespace {

// Host metadata (aclIntArray) is materialized as the int64 tensor the kernel input expects.
const aclTensor *ConvertMetadata(const aclIntArray *values, aclOpExecutor *executor)
{
    if (values == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor = executor->ConvertToTensor(values, DataType::DT_INT64);
    if (tensor == nullptr) {
        return nullptr;
    }
    auto *mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    return tensor;
}

} // namespace

const std::array<const aclTensor *, 1> ChunkFwdO(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *h,
    const aclTensor *g,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkOffsetsOptional,
    double scale,
    int64_t chunkSize,
    bool useExp2,
    bool stateVFirst,
    const char *outputLayout,
    const aclTensor *oOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkFwdO, q, k, v, h, g, cuSeqlensOptional, chunkOffsetsOptional, scale, chunkSize, useExp2, stateVFirst,
           outputLayout, oOut);

    std::string outputLayoutStr(outputLayout == nullptr ? "BNSD" : outputLayout);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ChunkFwdO,
        OP_INPUT(q, k, v, h, g, cuSeqlensOptional, chunkOffsetsOptional),
        OP_OUTPUT(oOut),
        OP_ATTR(scale, chunkSize, useExp2, stateVFirst, outputLayoutStr));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr};
    }
    return {oOut};
}

const std::array<const aclTensor *, 1> ChunkFwdO(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *h,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkOffsetsOptional,
    double scale,
    int64_t chunkSize,
    bool useExp2,
    bool stateVFirst,
    const char *outputLayout,
    const aclTensor *oOut,
    aclOpExecutor *executor)
{
    const aclTensor *cuSeqlens = ConvertMetadata(cuSeqlensOptional, executor);
    const aclTensor *chunkOffsets = ConvertMetadata(chunkOffsetsOptional, executor);
    if ((cuSeqlensOptional != nullptr && cuSeqlens == nullptr) ||
        (chunkOffsetsOptional != nullptr && chunkOffsets == nullptr)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert metadata to tensor failed.");
        return {nullptr};
    }
    return ChunkFwdO(q, k, v, h, g, cuSeqlens, chunkOffsets, scale, chunkSize, useExp2, stateVFirst,
                     outputLayout, oOut, executor);
}

} // namespace l0op
