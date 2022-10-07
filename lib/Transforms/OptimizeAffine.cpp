#include "PassDetail.hpp"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tvm-mlir/Support/Common.hpp"
#include "tvm-mlir/Transforms/Passes.hpp"

namespace mlir {

namespace {

struct FuseAffine : public OpRewritePattern<func::FuncOp> {
    FuseAffine(MLIRContext *ctx) : OpRewritePattern(ctx) {}

    LogicalResult matchAndRewrite(func::FuncOp func,
                                  PatternRewriter &rewriter) const override;
};

inline static uint32_t getPerfectlyNestedDepth(AffineForOp root) {
    SmallVector<AffineForOp> loops;
    getPerfectlyNestedLoops(loops, root);
    return loops.size();
}

LogicalResult FuseAffine::matchAndRewrite(func::FuncOp func,
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

class OptimizeAffine : public OptimizeAffineBase<OptimizeAffine> {
    void runOnOperation() override;
};

void OptimizeAffine::runOnOperation() {
    auto mod = getOperation();
    auto ctx = &getContext();
    {
        RewritePatternSet patterns(ctx);
        patterns.add<FuseAffine>(ctx);
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
