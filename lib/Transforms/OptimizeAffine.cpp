#include "PassDetail.hpp"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "relay-mlir/Support/Common.hpp"
#include "relay-mlir/Transforms/Passes.hpp"

namespace mlir {

namespace {

struct FuseLoop : public OpRewritePattern<func::FuncOp> {
    FuseLoop(MLIRContext *ctx) : OpRewritePattern(ctx) {}

    LogicalResult matchAndRewrite(func::FuncOp func,
                                  PatternRewriter &rewriter) const override;
};

inline static uint32_t getPerfectlyNestedDepth(AffineForOp root) {
    SmallVector<AffineForOp> loops;
    getPerfectlyNestedLoops(loops, root);
    return loops.size();
}

LogicalResult FuseLoop::matchAndRewrite(func::FuncOp func,
                                        PatternRewriter &rewriter) const {
    // Skip non-primitive function
    if (!func->hasAttrOfType<BoolAttr>("primitive")) return failure();

    // Collect all allocation and deallocation operations
    SmallVector<memref::AllocOp> allocOps;
    SmallVector<memref::DeallocOp> deallocOps;
    for (auto &op : func.getOps()) {
        if (isa<memref::AllocOp>(&op))
            allocOps.push_back(cast<memref::AllocOp>(&op));
        else if (isa<memref::DeallocOp>(&op))
            deallocOps.push_back(cast<memref::DeallocOp>(&op));
    }

    // Create a new function
    auto newFunc = rewriter.create<func::FuncOp>(func.getLoc(), func.getName(),
                                                 func.getFunctionType());
    for (auto &attr : func->getAttrs()) {
        if (llvm::is_contained(func.getAttributeNames(),
                               attr.getName().strref()))
            continue;
        newFunc->setAttr(attr.getName().strref(), attr.getValue());
    }
    rewriter.setInsertionPointToStart(newFunc.addEntryBlock());

    // Create value mapping
    BlockAndValueMapping mapper;
    for (auto [prevArg, newArg] :
         llvm::zip(func.getArguments(), newFunc.getArguments()))
        mapper.map(prevArg, newArg);

    // Reorder operations
    for (auto op : allocOps) rewriter.clone(*op, mapper);  // allocation first
    SmallVector<AffineForOp> forOps;
    for (auto &op : func.getOps()) {
        if (isa<memref::AllocOp, memref::DeallocOp>(&op)) continue;
        auto newOp = rewriter.clone(op, mapper);
        if (isa<AffineForOp>(newOp)) forOps.push_back(cast<AffineForOp>(newOp));
    }  // then other operations
    rewriter.setInsertionPoint(rewriter.getBlock()->getTerminator());
    for (auto op : deallocOps)
        rewriter.clone(*op, mapper);  // deallocation finally

    // Erase previous function
    rewriter.eraseOp(func);

    // Try fuse two consequent loops
    auto forIdx = 0u;
    while (forIdx + 1 < forOps.size()) {
        auto dstFor = forOps[forIdx], srcFor = forOps[forIdx + 1];
        ComputationSliceState srcSlice;
        auto dstDepth = getPerfectlyNestedDepth(dstFor);
        auto fuseResult = canFuseLoops(srcFor, dstFor, dstDepth, &srcSlice);
        if (fuseResult.value != FusionResult::Success) {
            forIdx++;
            continue;
        }
        fuseLoops(srcFor, dstFor, srcSlice, false);
        rewriter.eraseOp(srcFor);
        forOps.erase(forOps.begin() + forIdx + 1);
    }

    return success();
}

struct ParallelizeLoop : public OpRewritePattern<AffineForOp> {
    ParallelizeLoop(MLIRContext *ctx) : OpRewritePattern(ctx) {}

    LogicalResult matchAndRewrite(AffineForOp forOp,
                                  PatternRewriter &rewriter) const override;
};

LogicalResult ParallelizeLoop::matchAndRewrite(
    AffineForOp root, PatternRewriter &rewriter) const {
    // Skip if this loop is nested in another loop
    if (!isa<func::FuncOp>(root->getParentOp())) return failure();

    // Get all perfectly-nested, non-reduction loops
    SmallVector<AffineForOp> nestedLoops;
    getPerfectlyNestedLoops(nestedLoops, root);
    if (llvm::any_of(nestedLoops, [](AffineForOp op) {
            return op.getNumIterOperands() != 0;
        }))
        return failure();

    // Initialize parallel operation
    auto lbMaps = llvm::to_vector(llvm::map_range(
        nestedLoops, std::mem_fn(&AffineForOp::getLowerBoundMap)));
    auto ubMaps = llvm::to_vector(llvm::map_range(
        nestedLoops, std::mem_fn(&AffineForOp::getUpperBoundMap)));
    SmallVector<Value> lbArgs, ubArgs;
    for (auto forOp : nestedLoops) {
        auto lbs = forOp.getLowerBoundOperands(),
             ubs = forOp.getUpperBoundOperands();
        lbArgs.append(lbs.begin(), lbs.end());
        ubArgs.append(ubs.begin(), ubs.end());
    }
    auto steps = llvm::to_vector(
        llvm::map_range(nestedLoops, std::mem_fn(&AffineForOp::getStep)));
    auto parOp = rewriter.create<AffineParallelOp>(root.getLoc(), llvm::None,
                                                   llvm::None, lbMaps, lbArgs,
                                                   ubMaps, ubArgs, steps);

    // Clone body from innermost loop
    BlockAndValueMapping mapper;
    auto forIvs = llvm::map_range(nestedLoops,
                                  std::mem_fn(&AffineForOp::getInductionVar));
    for (auto [forIv, parIv] :
         llvm::zip(forIvs, parOp.getBody()->getArguments()))
        mapper.map(forIv, parIv);
    auto innerForOp = nestedLoops.back();
    rewriter.setInsertionPointToStart(parOp.getBody());
    for (auto &op : *innerForOp.getBody()) {
        if (&op != innerForOp.getBody()->getTerminator())
            rewriter.clone(op, mapper);
    }
    rewriter.replaceOp(root, parOp.getResults());

    return success();
}

struct VectorizeLoop : public OpRewritePattern<func::FuncOp> {
    VectorizeLoop(MLIRContext *ctx) : OpRewritePattern(ctx) {}

    LogicalResult matchAndRewrite(func::FuncOp func,
                                  PatternRewriter &rewriter) const override;
};

LogicalResult VectorizeLoop::matchAndRewrite(func::FuncOp func,
                                             PatternRewriter &rewriter) const {
    // Collect loops
    DenseSet<Operation *> loops;
    ReductionLoopMap reduceLoops;
    func.walk([&](AffineForOp forOp) {
        SmallVector<LoopReduction, 2> reductions;
        if (!isLoopParallel(forOp, &reductions)) return;
        loops.insert(forOp);
        if (!reductions.empty()) reduceLoops.insert({forOp, reductions});
    });

    // Perform vectorization
    if (loops.empty()) return failure();
    rewriter.updateRootInPlace(func, [&]() {
        vectorizeAffineLoops(func, loops, {8}, {}, reduceLoops);
    });
    return success();
}

class OptimizeAffine : public OptimizeAffineBase<OptimizeAffine> {
    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<AffineDialect, vector::VectorDialect>();
    }
};

void OptimizeAffine::runOnOperation() {
    // Get module and context
    auto mod = getOperation();
    auto ctx = &getContext();

    // Fuse consequent loops
    {
        RewritePatternSet patterns(ctx);
        patterns.add<FuseLoop>(ctx);
        auto funcs = llvm::to_vector(
            llvm::map_range(mod.getOps(), [](auto &op) { return &op; }));
        applyOpPatternsAndFold(funcs, std::move(patterns), true);
    }

    // Eliminate redundant load/store
    auto &domInfo = getAnalysis<DominanceInfo>();
    auto &postDomInfo = getAnalysis<PostDominanceInfo>();
    for (auto &op : mod.getOps())
        affineScalarReplace(cast<func::FuncOp>(&op), domInfo, postDomInfo);

    // Parallelize loops
    {
        RewritePatternSet patterns(ctx);
        patterns.add<ParallelizeLoop>(ctx);
        if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed())
            signalPassFailure();
    }

    // Vectorize loops
    {
        RewritePatternSet patterns(ctx);
        patterns.add<VectorizeLoop>(ctx);
        auto funcs = llvm::to_vector(
            llvm::map_range(mod.getOps(), [](auto &op) { return &op; }));
        applyOpPatternsAndFold(funcs, std::move(patterns), true);
    }
}

}  // namespace

std::unique_ptr<Pass> createOptimizeAffine() {
    return std::make_unique<OptimizeAffine>();
}

}  // namespace mlir
