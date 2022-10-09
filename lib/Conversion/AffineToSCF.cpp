#include "PassDetail.hpp"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class AffineToSCF : public AffineToSCFBase<AffineToSCF> {
    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<scf::SCFDialect>();
    }
};

void AffineToSCF::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithmeticDialect, BuiltinDialect,
                           memref::MemRefDialect, scf::SCFDialect>();
    target.addIllegalDialect<AffineDialect>();
    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
        signalPassFailure();
}

std::unique_ptr<Pass> createAffineToSCF() {
    return std::make_unique<AffineToSCF>();
}

}  // namespace mlir
