#include "PassDetail.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "relay-mlir/Conversion/Passes.hpp"
#include "relay-mlir/Dialect/Relay/RelayDialect.hpp"
#include "relay-mlir/Dialect/Relay/RelayOps.hpp"
#include "relay-mlir/Support/Common.hpp"

namespace mlir {

namespace {

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
            if (isa<func::ReturnOp>(owner)) {
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
            auto alloc = rewriter.create<memref::AllocOp>(
                op.getLoc(), result.getType().template cast<MemRefType>());
            newResult = alloc.getResult();
        }
        newResults.push_back(newResult);
    }

    auto buffers = llvm::to_vector(op->getOperands());
    buffers.append(newResults);

    // Lower operation with given buffers
    if (this->lower(op, buffers, rewriter).failed()) return failure();

    // Erase or replace previous operation
    if (!isFuncRet)
        rewriter.replaceOp(op, newResults);
    else
        rewriter.eraseOp(op);

    return success();
}

#define FOR(iv, low, high, body)                                          \
    auto iv##Loop = rewriter.create<AffineForOp>(op.getLoc(), low, high); \
    rewriter.setInsertionPointToStart(iv##Loop.getBody());                \
    {                                                                     \
        auto iv = iv##Loop.getInductionVar();                             \
        body                                                              \
    }                                                                     \
    rewriter.setInsertionPointAfter(iv##Loop);

#define LOAD(buffer, indices) \
    rewriter.create<AffineLoadOp>(op.getLoc(), buffer, indices).getResult()

#define STORE(value, buffer, indices) \
    rewriter.create<AffineStoreOp>(op.getLoc(), value, buffer, indices)

#define F32_CONST(value)                                                   \
    rewriter                                                               \
        .create<arith::ConstantFloatOp>(op.getLoc(), llvm::APFloat(value), \
                                        rewriter.getF32Type())             \
        .getResult()

#define BOP(Op, lhs, rhs) rewriter.create<Op>(op.getLoc(), lhs, rhs).getResult()

#define ADDF(lhs, rhs) BOP(arith::AddFOp, lhs, rhs)
#define MULF(lhs, rhs) BOP(arith::MulFOp, lhs, rhs)
#define MAXF(lhs, rhs) BOP(arith::MaxFOp, lhs, rhs)

inline static void genNestedLoops(
    Value result, PatternRewriter &rewriter,
    function_ref<void(const SmallVector<Value> &)> body) {
    auto shape = result.getType().cast<MemRefType>().getShape();
    SmallVector<Value> ivs;
    for (auto dim : shape) {
        auto loop = rewriter.create<AffineForOp>(result.getLoc(), 0, dim);
        ivs.push_back(loop.getInductionVar());
        rewriter.setInsertionPointToStart(loop.getBody());
    }
    body(ivs);
}

struct LowerReLU : public LowerOp<relay::ReLUOp> {
    LowerReLU(MLIRContext *ctx) : LowerOp(ctx) {}

    LogicalResult lower(relay::ReLUOp op, ValueRange buffers,
                        PatternRewriter &rewriter) const override {
        auto data = buffers[0], result = buffers[1];
        genNestedLoops(op.getResult(), rewriter,
                       [&](const SmallVector<Value> &ivs) {
                           auto x = LOAD(data, ivs);
                           auto y = MAXF(x, F32_CONST(0.f));
                           STORE(y, result, ivs);
                       });
        return success();
    }
};

struct LowerBiasAdd : public LowerOp<relay::BiasAddOp> {
    LowerBiasAdd(MLIRContext *ctx) : LowerOp(ctx) {}

    LogicalResult lower(relay::BiasAddOp op, ValueRange buffers,
                        PatternRewriter &rewriter) const override {
        auto data = buffers[0], bias = buffers[1], result = buffers[2];
        auto axis = op.getAxis();
        genNestedLoops(op.getResult(), rewriter,
                       [&](const SmallVector<Value> &ivs) {
                           auto x = LOAD(data, ivs);
                           auto b = LOAD(bias, (ValueRange{ivs[axis]}));
                           auto y = ADDF(x, b);
                           STORE(y, result, ivs);
                       });
        return success();
    }
};

struct LowerDense : public LowerOp<relay::DenseOp> {
    LowerDense(MLIRContext *ctx) : LowerOp(ctx) {}

    LogicalResult lower(relay::DenseOp op, ValueRange buffers,
                        PatternRewriter &rewriter) const override {
        auto data = buffers[0], weight = buffers[1], result = buffers[2];
        auto dataShape = data.getType().cast<MemRefType>().getShape();
        auto weightShape = weight.getType().cast<MemRefType>().getShape();
        auto batchSize = dataShape[0], inDim = dataShape[1],
             outDim = weightShape[0];

        FOR(i, 0, batchSize,  // for (i, 0, data.shape[0])
            FOR(
                j, 0, outDim,  // for (j, 0, weight.shape[0])
                auto kLoop = rewriter.create<AffineForOp>(
                    op.getLoc(), 0, inDim, 1, ValueRange{F32_CONST(0.f)});
                rewriter.setInsertionPointToStart(kLoop.getBody()); {
                    auto k = kLoop.getInductionVar();
                    auto D_ik = LOAD(data, (ValueRange{i, k}));
                    auto W_jk = LOAD(weight, (ValueRange{j, k}));
                    auto mul = MULF(D_ik, W_jk);
                    auto prev = kLoop.getRegionIterArgs()[0];
                    auto add = ADDF(prev, mul);
                    rewriter.create<AffineYieldOp>(op.getLoc(),
                                                   ValueRange{add});
                } rewriter.setInsertionPointAfter(kLoop);
                STORE(kLoop->getResult(0), result,
                      (ValueRange{i, j}));)  // end j
            )                                // end i

        return success();
    }
};

struct LowerCall : public LowerOp<func::CallOp> {
    LowerCall(MLIRContext *ctx) : LowerOp(ctx) {}

    LogicalResult lower(func::CallOp op, ValueRange buffers,
                        PatternRewriter &rewriter) const override {
        rewriter.create<func::CallOp>(
            op.getLoc(), rewriter.getStringAttr(op.getCallee() + "_lowered"),
            llvm::None, buffers);
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
    auto isMain = func.getName() == "main";
    auto inBufTypes = llvm::to_vector(
        llvm::map_range(func.getArgumentTypes(), cvtTensorToMemref));
    auto outBufTypes = llvm::to_vector(
        llvm::map_range(func.getResultTypes(), cvtTensorToMemref));
    auto bufTypes =
        llvm::to_vector(llvm::concat<Type>(inBufTypes, outBufTypes));
    auto newName = isMain ? func.getName() : func.getName() + "_lowered";
    auto newFunc = rewriter.create<func::FuncOp>(
        func.getLoc(), rewriter.getStringAttr(newName),
        rewriter.getFunctionType(bufTypes, llvm::None));
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
            if (isa<func::ReturnOp>(op)) continue;
            for (auto [prevArg, newArg] :
                 llvm::zip(op.getOperands(), newOp->getOperands())) {
                if (!lastUse.count(prevArg)) continue;
                if (lastUse[prevArg] != &op) continue;
                rewriter.create<memref::DeallocOp>(op.getLoc(), newArg);
            }
        }
    }

    // Replace original function
    if (isMain)
        rewriter.replaceOp(func, newFunc->getResults());
    else {
        rewriter.setInsertionPointAfter(func);
        rewriter.create<func::FuncOp>(
            func.getLoc(), func.getName(),
            rewriter.getFunctionType(inBufTypes, outBufTypes),
            rewriter.getStringAttr("private"));  // dummy function symbol
        rewriter.eraseOp(func);
    }

    return success();
}

struct EraseDummyFunc : public OpRewritePattern<func::FuncOp> {
    EraseDummyFunc(MLIRContext *ctx) : OpRewritePattern(ctx) {}

    LogicalResult matchAndRewrite(func::FuncOp func,
                                  PatternRewriter &rewriter) const override {
        if (func.getVisibility() == SymbolTable::Visibility::Private)
            rewriter.eraseOp(func);
        return success();
    }
};

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
        [](func::CallOp op) { return op->getNumResults() == 0; });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [](func::ReturnOp op) { return op.getNumOperands() == 0; });

    // Add rewrite patterns and apply conversion
    auto mod = getOperation();
    auto ctx = &getContext();
    {
        RewritePatternSet patterns(ctx);
        patterns.add<LowerFunc, LowerReLU, LowerBiasAdd, LowerDense, LowerCall,
                     EraseReturnValue>(ctx);
        if (applyPartialConversion(mod, target, std::move(patterns)).failed())
            signalPassFailure();
    }
    {
        RewritePatternSet patterns(ctx);
        patterns.add<EraseDummyFunc>(ctx);
        auto funcs = llvm::to_vector(
            llvm::map_range(mod.getOps(), [](auto &op) { return &op; }));
        applyOpPatternsAndFold(funcs, std::move(patterns), true);
    }
}

}  // namespace

std::unique_ptr<Pass> createRelayToAffine() {
    return std::make_unique<RelayToAffine>();
}

}  // namespace mlir
