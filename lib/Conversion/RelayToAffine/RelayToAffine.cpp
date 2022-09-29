#include "tvm-mlir/Conversion/RelayToAffine.hpp"

#include "../PassDetail.hpp"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
#include "tvm-mlir/Support/Common.hpp"

namespace mlir {

inline static Type cvtTensorToMemref(Type type) {
    auto tt = type.cast<TensorType>();
    return MemRefType::get(tt.getShape(), tt.getElementType());
}

struct LowerReLU : public OpRewritePattern<relay::ReLUOp> {
    LowerReLU(MLIRContext *ctx) : OpRewritePattern(ctx) {}

    LogicalResult matchAndRewrite(relay::ReLUOp op,
                                  PatternRewriter &rewriter) const override {
        return success();
    }
};

struct LowerBiasAdd : public OpRewritePattern<relay::BiasAddOp> {
    LowerBiasAdd(MLIRContext *ctx) : OpRewritePattern(ctx) {}

    LogicalResult matchAndRewrite(relay::BiasAddOp op,
                                  PatternRewriter &rewriter) const override {
        return success();
    }
};

struct LowerDense : public OpRewritePattern<relay::DenseOp> {
    LowerDense(MLIRContext *ctx) : OpRewritePattern(ctx) {}

    LogicalResult matchAndRewrite(relay::DenseOp op,
                                  PatternRewriter &rewriter) const override {
        return success();
    }
};

struct LowerCall : public OpRewritePattern<func::CallOp> {
    LowerCall(MLIRContext *ctx) : OpRewritePattern(ctx) {}

    LogicalResult matchAndRewrite(func::CallOp op,
                                  PatternRewriter &rewriter) const override {
        // Find buffers for function outputs
        SmallVector<Value> newResults;
        bool isFuncRet = false;
        for (auto result : op->getResults()) {
            Value newResult;
            func::ReturnOp retOp;
            int64_t retIdx = -1;
            for (auto &use : result.getUses()) {
                auto owner = use.getOwner();
                if (owner->getName().getStringRef() ==
                    func::ReturnOp::getOperationName()) {
                    retOp = cast<func::ReturnOp>(owner);
                    retIdx = use.getOperandNumber();
                    isFuncRet = true;
                }
            }
            if (retOp) {
                auto func = cast<func::FuncOp>(op->getParentOp());
                auto numInputs =
                    func->getAttrOfType<IntegerAttr>("num_inputs").getInt();
                newResult = func.getArgument(numInputs + retIdx);
            } else {
                auto alloca = rewriter.create<memref::AllocaOp>(
                    op.getLoc(), result.getType().cast<MemRefType>());
                newResult = alloca.getResult();
            }
            newResults.push_back(newResult);
        }
        auto buffers = llvm::to_vector(op.getOperands());
        buffers.append(newResults);

        // Lower call operation
        rewriter.create<func::CallOp>(op.getLoc(), op.getCallee(), llvm::None,
                                      buffers);

        // Erase or replace previous operations
        if (isFuncRet)
            rewriter.eraseOp(op);
        else
            rewriter.replaceOp(op, newResults);

        return success();
    }
};

struct EraseReturnValue : public OpRewritePattern<func::ReturnOp> {
    EraseReturnValue(MLIRContext *ctx) : OpRewritePattern(ctx) {}

    LogicalResult matchAndRewrite(func::ReturnOp op,
                                  PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, llvm::None);
        return success();
    }
};

struct LowerFunc : public OpRewritePattern<func::FuncOp> {
    LowerFunc(MLIRContext *ctx) : OpRewritePattern(ctx) {}

    LogicalResult matchAndRewrite(func::FuncOp func,
                                  PatternRewriter &rewriter) const override;
};

LogicalResult LowerFunc::matchAndRewrite(func::FuncOp func,
                                         PatternRewriter &rewriter) const {
    // Convert function prototype
    Debug("{}", func.getName());
    auto inTypes = llvm::to_vector(
        llvm::map_range(func.getArgumentTypes(), cvtTensorToMemref));
    inTypes.append(llvm::to_vector(
        llvm::map_range(func.getResultTypes(), cvtTensorToMemref)));
    auto newFunc = rewriter.create<func::FuncOp>(
        func.getLoc(), func.getName(),
        rewriter.getFunctionType(inTypes, llvm::None));
    if (func->hasAttrOfType<BoolAttr>("primitive"))
        newFunc->setAttr("primitive", rewriter.getBoolAttr(true));
    newFunc->setAttr("num_inputs",
                     rewriter.getI64IntegerAttr(func.getNumArguments()));

    // Convert operations in the function
    rewriter.setInsertionPointToStart(newFunc.addEntryBlock());
    BlockAndValueMapping mapper;
    for (auto [tValue, mValue] :
         llvm::zip(func.getArguments(), newFunc.getArguments()))
        mapper.map(tValue, mValue);
    for (auto &block : func.getRegion()) {
        for (auto &op : block) {
            auto newOp = rewriter.clone(op, mapper);
            for (auto result : newOp->getResults())
                result.setType(cvtTensorToMemref(result.getType()));
        }
    }
    rewriter.eraseOp(func);

    return success();
}

class RelayToAffine : public RelayToAffineBase<RelayToAffine> {
    void runOnOperation() override;
};

void RelayToAffine::runOnOperation() {
    // Define conversion target
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, arith::ArithmeticDialect,
                           BuiltinDialect, func::FuncDialect,
                           memref::MemRefDialect>();
    target.addIllegalDialect<relay::RelayDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp func) {
        return llvm::none_of(func.getArgumentTypes(),
                             [](Type type) { return type.isa<TensorType>(); });
    });
    target.addDynamicallyLegalOp<func::CallOp>(
        [](func::CallOp op) { return op.getNumResults() == 0; });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [](func::ReturnOp op) { return op.getNumOperands() == 0; });

    // Add rewrite patterns
    auto ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerFunc, LowerReLU, LowerBiasAdd, LowerDense, LowerCall,
                 EraseReturnValue>(ctx);

    // Apply conversion
    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
        signalPassFailure();
}

std::unique_ptr<Pass> createRelayToAffine() {
    return std::make_unique<RelayToAffine>();
}

}  // namespace mlir
