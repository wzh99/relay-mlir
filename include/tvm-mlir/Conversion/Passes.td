#ifndef TVM_MLIR_CONVERSION_PASSES
#define TVM_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def RelayToAffine : Pass<"relay-to-affine", "ModuleOp"> {
    let summary = "Lower Relay to Affine.";
    let description = [{
    Convert Relay operations to low-level implementation in Affine dialect.
    }];

    let constructor = "createRelayToAffine()";
}

def AffineToSCF : Pass<"affine-to-scf", "ModuleOp"> {
    let summary = "Lower Affine to SCF.";
    let description = [{
    Lower loops in Affine dialect to SCF dialect.
    }];

    let constructor = "createAffineToSCF()";
}

def SCFToLLVM : Pass<"scf-to-llvm", "ModuleOp"> {
    let summary = "Lower SCF to LLVM.";
    let description = [{
    Lower loops in SCF dialect to LLVM IR.
    }];

    let constructor = "createSCFToLLVM()";
}

#endif // TVM_MLIR_CONVERSION_PASSES
