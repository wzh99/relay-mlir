#include "PassDetail.hpp"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "tvm-mlir/Conversion/Passes.hpp"

namespace mlir {

class AffineToLLVM : public AffineToLLVMBase<AffineToLLVM> {
    void runOnOperation() override;
};

void AffineToLLVM::runOnOperation() {
    // Define conversion target
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    LLVMTypeConverter converter(&getContext());

    // Populate conversion patterns
    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
    arith::populateArithmeticExpandOpsPatterns(patterns);
    populateMemRefToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);

    // Completely lower to LLVM IR
    if (applyFullConversion(getOperation(), target, std::move(patterns))
            .failed())
        signalPassFailure();
}

std::unique_ptr<Pass> createAffineToLLVM() {
    return std::make_unique<AffineToLLVM>();
}

}  // namespace mlir
