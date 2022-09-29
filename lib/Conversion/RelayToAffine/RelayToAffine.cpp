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

template <class Op>
struct LowerOp : public OpRewritePattern<Op> {
    LowerOp(MLIRContext *ctx) : OpRewritePattern<Op>(ctx) {}

    LogicalResult matchAndRewrite(Op op,
                                  PatternRewriter &rewriter) const override;

    virtual LogicalResult lower(Op op, ValueRange buffers,
                                PatternRewriter &rewriter) const = 0;
};

template <class Op>
LogicalResult LowerOp<Op>::matchAndRewrite(Op op,
                                           PatternRewriter &rewriter) const {
    // Find buffers for function outputs
    SmallVector<Value> newResults;
    bool isFuncRet = false;

    for (auto result : op->getResults()) {
        // Check if the result of this operation is returned by the parent
        // function
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

        // Collect result buffer or allocate a new one
        if (retOp) {
            auto func = cast<func::FuncOp>(op->getParentOp());
            auto numInputs =
                func->template getAttrOfType<IntegerAttr>("num_inputs")
                    .getInt();
            newResult = func.getArgument(numInputs + retIdx);
        } else {
            auto alloca = rewriter.create<memref::AllocaOp>(
                op.getLoc(), result.getType().template cast<MemRefType>());
            newResult = alloca.getResult();
        }
        newResults.push_back(newResult);
    }

    auto buffers = llvm::to_vector(op.getOperands());
    buffers.append(newResults);

    // Lower operation with given buffers
    if (this->lower(op, buffers, rewriter).failed()) return failure();

    // Erase or replace previous operations
    if (isFuncRet)
        op.erase();
    else
        rewriter.replaceOp(op, newResults);

    return success();
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

#define FOR(name, low, high, body)                                         \
    auto name##For = rewriter.create<AffineForOp>(op.getLoc(), low, high); \
    rewriter.setInsertionPointToStart(name##For.getBody());                \
    {                                                                      \
        auto name##Iv = name##For.getInductionVar();                       \
        body                                                               \
    }                                                                      \
    rewriter.setInsertionPoint(name##For.getBody()->getTerminator());

#define LOAD(buffer, indices) \
    rewriter.create<AffineLoadOp>(op.getLoc(), buffer, indices).getResult()

#define STORE(value, buffer, indices) \
    rewriter.create<AffineStoreOp>(op.getLoc(), value, buffer, indices)

#define F32_CONST(value)                                                   \
    rewriter                                                               \
        .create<arith::ConstantFloatOp>(op.getLoc(), llvm::APFloat(value), \
                                        rewriter.getF32Type())             \
        .getResult();

#define BOP(Op, lhs, rhs) rewriter.create<Op>(op.getLoc(), lhs, rhs).getResult()

#define ADDF(lhs, rhs) BOP(arith::AddFOp, lhs, rhs)
#define MULF(lhs, rhs) BOP(arith::MulFOp, lhs, rhs)

struct LowerDense : public LowerOp<relay::DenseOp> {
    LowerDense(MLIRContext *ctx) : LowerOp(ctx) {}

    LogicalResult lower(relay::DenseOp op, ValueRange buffers,
                        PatternRewriter &rewriter) const override {
        auto data = buffers[0], weight = buffers[1], result = buffers[2];
        auto dataShape = data.getType().cast<MemRefType>().getShape();
        auto weightShape = weight.getType().cast<MemRefType>().getShape();
        auto batchSize = dataShape[0], inDim = dataShape[1],
             outDim = weightShape[0];

        FOR(batch, 0, batchSize,  // for (i, 0, data.shape[0])
            FOR(out, 0, outDim,   // for (j, 0, weight.shape[0])
                auto init = F32_CONST(1.f);
                STORE(init, result, (ValueRange{batchIv, outIv}));
                FOR(in, 0, inDim,  // for (k, 0, data.shape[i])
                    auto D_ik =
                        LOAD(data, (ValueRange{batchIv, inIv}));  // data[i, k]
                    auto W_jk = LOAD(
                        weight, (ValueRange{outIv, inIv}));  // weight[j, k]
                    auto mul = MULF(D_ik, W_jk);
                    auto prev = LOAD(result, (ValueRange{batchIv, outIv}));
                    auto add = ADDF(prev, mul);
                    // result[i, j] += data[i, k] * weight[j, k]
                    STORE(add, result, (ValueRange{batchIv, outIv}));)  // end k
                )                                                       // end j
            )                                                           // end i

        return success();
    }
};

struct LowerCall : public LowerOp<func::CallOp> {
    LowerCall(MLIRContext *ctx) : LowerOp(ctx) {}

    LogicalResult lower(func::CallOp op, ValueRange buffers,
                        PatternRewriter &rewriter) const override {
        rewriter.create<func::CallOp>(op.getLoc(), op.getCallee(), llvm::None,
                                      buffers);
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

    // Find the last use of each intermediate value
    DenseMap<Value, Operation *> lastUse;
    for (auto &block : func.getRegion()) {
        for (auto &op : block) {
            for (auto arg : op.getOperands())
                if (lastUse.count(arg)) lastUse[arg] = &op;
            for (auto result : op.getResults())
                lastUse.insert({result, nullptr});
        }
    }

    // Convert operations in the function
    rewriter.setInsertionPointToStart(newFunc.addEntryBlock());
    BlockAndValueMapping mapper;
    for (auto [tValue, mValue] :
         llvm::zip(func.getArguments(), newFunc.getArguments()))
        mapper.map(tValue, mValue);
    for (auto &block : func.getRegion()) {
        for (auto &op : block) {
            // Clone operation and set result types
            auto newOp = rewriter.clone(op, mapper);
            for (auto result : newOp->getResults())
                result.setType(cvtTensorToMemref(result.getType()));

            // Deallocate arguments which are lastly used by this operation
            if (op.getName().getStringRef() ==
                func::ReturnOp::getOperationName())
                continue;
            for (auto [prevArg, newArg] :
                 llvm::zip(op.getOperands(), newOp->getOperands())) {
                if (!lastUse.count(prevArg)) continue;
                if (lastUse[prevArg] != &op) continue;
                rewriter.create<memref::DeallocOp>(op.getLoc(), newArg);
            }
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
