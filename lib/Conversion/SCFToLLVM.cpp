#include "PassDetail.hpp"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "relay-mlir/Conversion/Passes.hpp"

namespace mlir {

namespace {

class SCFToLLVM : public SCFToLLVMBase<SCFToLLVM> {
    void runOnOperation() override;
};

void SCFToLLVM::runOnOperation() {
    // Progressive lowering inside dialects
    {
        RewritePatternSet patterns(&getContext());
        arith::populateArithmeticExpandOpsPatterns(patterns);
        vector::populateVectorToVectorCanonicalizationPatterns(patterns);
        vector::populateVectorTransferLoweringPatterns(patterns, 1);
        vector::populateVectorMaskMaterializationPatterns(patterns, false);
        applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
            .succeeded();
    }

    // Convert to LLVM Dialect
    {
        LLVMConversionTarget target(getContext());
        target.addLegalOp<ModuleOp>();
        LLVMTypeConverter converter(&getContext());
        configureOpenMPToLLVMConversionLegality(target, converter);
        target.addLegalOp<scf::YieldOp, omp::YieldOp, omp::TerminatorOp>();

        RewritePatternSet patterns(&getContext());
        populateSCFToControlFlowConversionPatterns(patterns);
        populateFuncToLLVMConversionPatterns(converter, patterns);
        populateOpenMPToLLVMConversionPatterns(converter, patterns);
        cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
        arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
        populateMemRefToLLVMConversionPatterns(converter, patterns);
        populateVectorToLLVMConversionPatterns(converter, patterns, true,
                                               false);
        vector::populateVectorMaskMaterializationPatterns(patterns, false);
        vector::populateVectorTransferLoweringPatterns(patterns, 1);

        // Completely lower to LLVM dialect
        if (applyFullConversion(getOperation(), target, std::move(patterns))
                .failed())
            signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<Pass> createSCFToLLVM() {
    return std::make_unique<SCFToLLVM>();
}

}  // namespace mlir
