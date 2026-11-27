/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkGatedDeltaRuleFwd);

const std::array<const aclTensor *, 4> ChunkGatedDeltaRuleFwd(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *aStorage,
    const aclTensor *rawG,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOptional,
    bool outputFinalState,
    int64_t chunkSize,
    double scale,
    bool outputGCumsum,
    const aclTensor *oOut,
    const aclTensor *finalStateOut,
    const aclTensor *gCumsumBthOut,
    const aclTensor *aOut,
    int64_t rawGLayout,
    int64_t qkvLayout,
    int64_t oLayout,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRuleFwd, q, k, v, beta, aStorage, rawG, gkOptional, initialStateOptional,
           cuSeqlensOptional, chunkIndicesOptional, outputFinalState, chunkSize, scale, outputGCumsum,
           oOut, finalStateOut, gCumsumBthOut, aOut, rawGLayout, qkvLayout, oLayout);

    const aclError ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleFwd,
        OP_INPUT(q, k, v, beta, aStorage, rawG, gkOptional, initialStateOptional,
                 cuSeqlensOptional, chunkIndicesOptional),
        OP_OUTPUT(oOut, finalStateOut, gCumsumBthOut, aOut),
        OP_ATTR(outputFinalState, chunkSize, scale, outputGCumsum, rawGLayout, qkvLayout, oLayout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkGatedDeltaRuleFwd failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {oOut, finalStateOut, gCumsumBthOut, aOut};
}

} // namespace l0op
