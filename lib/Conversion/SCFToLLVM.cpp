#include "PassDetail.hpp"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "tvm-mlir/Conversion/Passes.hpp"

namespace mlir {

namespace {

class SCFToLLVM : public SCFToLLVMBase<SCFToLLVM> {
    void runOnOperation() override;
};

void SCFToLLVM::runOnOperation() {
    // Define conversion target
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    LLVMTypeConverter converter(&getContext());
    configureOpenMPToLLVMConversionLegality(target, converter);
    target.addLegalOp<scf::YieldOp, omp::YieldOp, omp::TerminatorOp>();

    // Populate conversion patterns
    RewritePatternSet patterns(&getContext());
    populateSCFToControlFlowConversionPatterns(patterns);
    populateOpenMPToLLVMConversionPatterns(converter, patterns);
    arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
    arith::populateArithmeticExpandOpsPatterns(patterns);
    populateMemRefToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);

    // Completely lower to LLVM dialect
    if (applyFullConversion(getOperation(), target, std::move(patterns))
            .failed())
        signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createSCFToLLVM() {
    return std::make_unique<SCFToLLVM>();
}

}  // namespace mlir
