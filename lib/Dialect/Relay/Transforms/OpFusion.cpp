#include <unordered_map>

#include "PassDetail.hpp"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tvm-mlir/Dialect/Relay/Passes.hpp"
#include "tvm-mlir/Support/Common.hpp"

namespace mlir {
namespace relay {

static LogicalResult fuseDenseBiasAddRelu(Operation *pivot,
                                          SmallVector<Operation *> &group,
                                          SmallVector<Operation *> &outputs) {
    // Match dense
    if (pivot->getName().getStringRef() != "relay.nn.dense") return failure();

    // Match bias_add
    auto denseResult = pivot->getResult(0);
    if (!denseResult.hasOneUse()) return failure();
    auto biasAdd = llvm::to_vector(denseResult.getUsers())[0];
    if (biasAdd->getName().getStringRef() != "relay.nn.bias_add")
        return failure();
    group = {pivot, biasAdd};
    outputs = {biasAdd};

    // Optionally match relu
    auto biasAddResult = biasAdd->getResult(0);
    if (!biasAddResult.hasOneUse()) return success();
    auto relu = llvm::to_vector(biasAddResult.getUsers())[0];
    if (relu->getName().getStringRef() != "relay.nn.relu") return success();
    group.push_back(relu);
    outputs = {relu};

    return success();
}

using MatchFn = LogicalResult (*)(Operation *, SmallVector<Operation *> &,
                                  SmallVector<Operation *> &);
static MatchFn matchFuncs[] = {fuseDenseBiasAddRelu};

struct FusionGroup {
    SmallVector<Operation *> ops;
    SmallVector<Operation *> outputs;
};

class OpFusionPattern : public RewritePattern {
public:
    OpFusionPattern(MLIRContext *ctx, const std::vector<FusionGroup> &groups,
                    const std::unordered_map<Operation *, size_t> &opGrpIdx)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), 1, ctx),
          groups(groups),
          opGrpIdx(opGrpIdx) {}

    LogicalResult matchAndRewrite(Operation *op,
                                  PatternRewriter &rewriter) const override {
        // Find out the group
        Debug("{}", op->getName().getStringRef());
        return success();
    }

private:
    const std::vector<FusionGroup> &groups;
    const std::unordered_map<Operation *, size_t> &opGrpIdx;
};

class OpFusion : public OpFusionBase<OpFusion> {
    void runOnOperation() override;
};

void OpFusion::runOnOperation() {
    // Get main function
    auto ctx = &getContext();
    auto mainFn = getOperation();
    if (mainFn.getName() != "main") return;

    // Match and fuse patterns
    SmallVector<Operation *> groupOps;
    SmallVector<Operation *> outputs;
    std::vector<FusionGroup> groups;
    std::unordered_map<Operation *, size_t> opGrpIdx;

    mainFn.walk([&](Operation *op) {
        // Skip operations not interested
        if (op->getDialect()->getNamespace() != "relay") return;
        if (opGrpIdx.count(op)) return;

        for (auto matcher : matchFuncs) {
            // Match with predefined functions
            if (matcher(op, groupOps, outputs).failed()) continue;
            assert(!groupOps.empty() && !outputs.empty());

            // Create fusion group
            FusionGroup group;
            group.ops.swap(groupOps);
            group.outputs.swap(outputs);
            auto grpIdx = groups.size();
            for (auto gOp : group.ops) opGrpIdx.insert({gOp, grpIdx});
            groups.push_back(std::move(group));
            return;
        }

        // Create single-operation group
        opGrpIdx.insert({op, groups.size()});
        groups.push_back({.ops = {op}, .outputs = {op}});
    });

    // Create nested function for each group
    RewritePatternSet patterns(ctx);
    patterns.add<OpFusionPattern>(ctx, groups, opGrpIdx);
    FrozenRewritePatternSet frozenPat(std::move(patterns));
    applyPatternsAndFoldGreedily(mainFn, frozenPat, {.maxIterations = 0})
        .succeeded();
}

std::unique_ptr<Pass> createOpFusion() { return std::make_unique<OpFusion>(); }

}  // namespace relay
}  // namespace mlir
