#include <unordered_map>

#include "PassDetail.hpp"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tvm-mlir/Dialect/Relay/Passes.hpp"
#include "tvm-mlir/Support/Common.hpp"

namespace mlir {
namespace relay {

static LogicalResult fuseDenseBiasAddRelu(Operation *pivot,
                                          SmallVector<Operation *> &group) {
    // Match dense
    if (pivot->getName().getStringRef() != "relay.nn.dense") return failure();

    // Match bias_add
    auto denseResult = pivot->getResult(0);
    if (!denseResult.hasOneUse()) return failure();
    auto biasAdd = llvm::to_vector(denseResult.getUsers())[0];
    if (biasAdd->getName().getStringRef() != "relay.nn.bias_add")
        return failure();
    group = {pivot, biasAdd};

    // Optionally match relu
    auto biasAddResult = biasAdd->getResult(0);
    if (!biasAddResult.hasOneUse()) return success();
    auto relu = llvm::to_vector(biasAddResult.getUsers())[0];
    if (relu->getName().getStringRef() != "relay.nn.relu") return success();
    group.push_back(relu);

    return success();
}

using MatchFn = LogicalResult (*)(Operation *, SmallVector<Operation *> &);
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

    LogicalResult matchAndRewrite(Operation *root,
                                  PatternRewriter &rewriter) const override;

private:
    mutable size_t nGroup = 0;
    const std::vector<FusionGroup> &groups;
    const std::unordered_map<Operation *, size_t> &opGrpIdx;
};

LogicalResult OpFusionPattern::matchAndRewrite(
    Operation *root, PatternRewriter &rewriter) const {
    // Find out the group
    if (root->getDialect()->getNamespace() != "relay") return success();
    if (cast<func::FuncOp>(root->getParentOp()).getName() != "main")
        return success();
    auto &group = groups[opGrpIdx.at(root)];

    // Only rewrite at outputs
    if (!llvm::is_contained(group.outputs, root)) return success();

    // Find all arguments of the new function
    DenseSet<Operation *> opSet(group.ops.begin(), group.ops.end());
    std::vector<Value> args;
    for (auto op : group.ops)
        for (auto arg : op->getOperands())
            if (!opSet.contains(arg.getDefiningOp())) args.push_back(arg);

    // Find all results of the new function
    std::vector<Value> results;
    std::vector<std::pair<size_t, size_t>> resultIndices;
    for (auto outOpZip : llvm::enumerate(group.outputs)) {
        for (auto resultZip : llvm::enumerate(outOpZip.value()->getResults())) {
            auto result = resultZip.value();
            if (llvm::any_of(result.getUsers(), [&](Operation *op) {
                    return !opSet.contains(op);
                })) {
                results.push_back(result);
                resultIndices.push_back({outOpZip.index(), resultZip.index()});
            }
        }
    }

    // Create prototype of the function
    auto inTypes = llvm::to_vector(
        llvm::map_range(args, [](Value in) { return in.getType(); }));
    auto outTypes = llvm::to_vector(
        llvm::map_range(results, [](Value out) { return out.getType(); }));
    auto funcType = rewriter.getFunctionType(inTypes, outTypes);
    auto funcName = fmt::format("fused_{}", nGroup++);
    rewriter.setInsertionPointToEnd(root->getParentOp()->getBlock());
    auto func =
        rewriter.create<func::FuncOp>(root->getLoc(), funcName, funcType);
    func->setAttr("primitive", rewriter.getBoolAttr(true));

    // Create function body
    auto block = func.addEntryBlock();
    BlockAndValueMapping mapper;
    for (auto [arg, param] : llvm::zip(args, block->getArguments()))
        mapper.map(arg, param);
    rewriter.setInsertionPointToStart(block);
    std::vector<Operation *> funcOutputs;
    for (auto op : group.ops) {
        auto clonedOp = rewriter.clone(*op, mapper);
        if (llvm::is_contained(group.outputs, op))
            funcOutputs.push_back(clonedOp);
    }
    SmallVector<Value> funcResults;
    for (auto [i, j] : resultIndices)
        funcResults.push_back(funcOutputs[i]->getResult(j));
    rewriter.create<func::ReturnOp>(root->getLoc(), funcResults);

    // Replace group with function call
    rewriter.setInsertionPointAfter(root);
    auto funcCall = rewriter.create<func::CallOp>(
        root->getLoc(), FlatSymbolRefAttr::get(func), outTypes, args);

    // Replace uses of group outputs
    auto callResults = funcCall.getResults();
    auto indexIter = resultIndices.begin();
    for (auto opZip : llvm::enumerate(group.outputs)) {
        auto begin = indexIter;
        while (indexIter->first == opZip.index()) ++indexIter;
        if (begin == indexIter) continue;
        SmallVector<Value> newValues;
        for (auto [_, j] : llvm::iterator_range(begin, indexIter))
            newValues.push_back(callResults[j]);
        rewriter.replaceOp(opZip.value(), newValues);
    }

    return success();
}

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
    std::vector<FusionGroup> groups;
    std::unordered_map<Operation *, size_t> opGrpIdx;

    mainFn.walk([&](Operation *pivot) {
        // Skip operations not interested
        if (pivot->getDialect()->getNamespace() != "relay") return;
        if (opGrpIdx.count(pivot)) return;

        for (auto matcher : matchFuncs) {
            // Match with predefined functions
            if (matcher(pivot, groupOps).failed()) continue;
            assert(!groupOps.empty());

            // Create fusion group
            FusionGroup group;
            group.ops.swap(groupOps);
            DenseSet<Operation *> opSet(group.ops.begin(), group.ops.end());
            for (auto op : group.ops) {
                auto isOut = false;
                for (auto result : op->getResults())
                    for (auto user : result.getUsers())
                        if (!opSet.contains(user)) isOut = true;
                if (isOut) group.outputs.push_back(op);
            }
            auto grpIdx = groups.size();
            for (auto op : group.ops) opGrpIdx.insert({op, grpIdx});
            groups.push_back(std::move(group));
            return;
        }

        // Create single-operation group
        opGrpIdx.insert({pivot, groups.size()});
        groups.push_back({.ops = {pivot}, .outputs = {pivot}});
    });

    // Create nested function for each group
    RewritePatternSet patterns(ctx);
    patterns.add<OpFusionPattern>(ctx, groups, opGrpIdx);
    FrozenRewritePatternSet frozenPat(std::move(patterns));
    GreedyRewriteConfig config{.useTopDownTraversal = true, .maxIterations = 0};
    applyPatternsAndFoldGreedily(mainFn, frozenPat, std::move(config))
        .succeeded();
}

std::unique_ptr<Pass> createOpFusion() { return std::make_unique<OpFusion>(); }

}  // namespace relay
}  // namespace mlir
