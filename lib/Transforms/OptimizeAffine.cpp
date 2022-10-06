#include "PassDetail.hpp"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tvm-mlir/Support/Common.hpp"
#include "tvm-mlir/Transforms/Passes.hpp"

namespace mlir {

namespace {

struct ReorderMemory : public OpRewritePattern<func::FuncOp> {
    ReorderMemory(MLIRContext *ctx) : OpRewritePattern(ctx) {}

    LogicalResult matchAndRewrite(func::FuncOp func,
                                  PatternRewriter &rewriter) const override;
};

LogicalResult ReorderMemory::matchAndRewrite(func::FuncOp func,
                                             PatternRewriter &rewriter) const {
    // Skip non-primitive function
    if (!func->hasAttrOfType<BoolAttr>("primitive")) return success();

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
    for (auto &op : func.getOps()) {
        if (isa<memref::AllocOp, memref::DeallocOp>(&op)) continue;
        rewriter.clone(op, mapper);
    }  // then other operations
    rewriter.setInsertionPoint(rewriter.getBlock()->getTerminator());
    for (auto op : deallocOps)
        rewriter.clone(*op, mapper);  // deallocation finally

    // Erase previous function
    rewriter.eraseOp(func);

    return success();
}

class OptimizeAffine : public OptimizeAffineBase<OptimizeAffine> {
    void runOnOperation() override;
};

void OptimizeAffine::runOnOperation() {
    // Reorder memory allocation and deallocation
    auto mod = getOperation();
    auto ctx = &getContext();
    {
        RewritePatternSet patterns(ctx);
        patterns.add<ReorderMemory>(ctx);
        auto funcs = llvm::to_vector(
            llvm::map_range(mod.getOps(), [](Operation &op) { return &op; }));
        applyOpPatternsAndFold(funcs, std::move(patterns), true);
    }
}

}  // namespace

std::unique_ptr<Pass> createOptimizeAffine() {
    return std::make_unique<OptimizeAffine>();
}

}  // namespace mlir
