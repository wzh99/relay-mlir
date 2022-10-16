#include <unordered_map>

#include "PassDetail.hpp"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "relay-mlir/Dialect/Relay/Passes.hpp"
#include "relay-mlir/Support/Common.hpp"

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
    if (root->getDialect()->getNamespace() !=
        RelayDialect::getDialectNamespace())
        return failure();
    if (cast<func::FuncOp>(root->getParentOp()).getName() != "main")
        return failure();
    auto &group = groups[opGrpIdx.at(root)];

    // Only rewrite at outputs
    if (!llvm::is_contained(group.outputs, root)) return failure();

    // Find all arguments of the new function
    DenseSet<Operation *> opSet(group.ops.begin(), group.ops.end());
    SmallVector<Value> args;
    for (auto op : group.ops)
        for (auto arg : op->getOperands())
            if (!opSet.contains(arg.getDefiningOp())) args.push_back(arg);

    // Find all results of the new function
    SmallVector<Value> results;
    for (auto outOp : group.outputs) {
        auto opResults = outOp->getResults();
        results.append(opResults.begin(), opResults.end());
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
    SmallVector<Operation *> funcOutOps;
    for (auto op : group.ops) {
        auto clonedOp = rewriter.clone(*op, mapper);
        if (llvm::is_contained(group.outputs, op))
            funcOutOps.push_back(clonedOp);
    }
    SmallVector<Value> funcResults;
    for (auto outOp : funcOutOps) {
        auto opResults = outOp->getResults();
        funcResults.append(opResults.begin(), opResults.end());
    }
    rewriter.create<func::ReturnOp>(root->getLoc(), funcResults);

    // Replace group with function call
    rewriter.setInsertionPointAfter(root);
    auto funcCall = rewriter.create<func::CallOp>(
        root->getLoc(), FlatSymbolRefAttr::get(func), outTypes, args);

    // Replace uses of group outputs
    auto resultIter = funcCall.getResults().begin();
    for (auto op : group.outputs) {
        SmallVector<Value> newValues(resultIter,
                                     resultIter + op->getNumResults());
        rewriter.replaceOp(op, newValues);
        resultIter += op->getNumResults();
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
    GreedyRewriteConfig config{.useTopDownTraversal = true};
    if (applyPatternsAndFoldGreedily(mainFn, std::move(patterns),
                                     std::move(config))
            .failed())
        signalPassFailure();
}

std::unique_ptr<Pass> createOpFusion() { return std::make_unique<OpFusion>(); }

}  // namespace relay
}  // namespace mlir
