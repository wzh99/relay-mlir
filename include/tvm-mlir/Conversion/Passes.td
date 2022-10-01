#ifndef TVM_MLIR_CONVERSION_PASSES
#define TVM_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def RelayToAffine : Pass<"relay-to-affine", "ModuleOp"> {
    let summary = "Lower Relay to Affine.";
    let description = [{
    Convert Relay operations to low-level implementation in Affine dialect.
    }];

    let constructor = "createRelayToAffine()";
    let dependentDialects = [
        "func::FuncDialect",
        "relay::RelayDialect",
        "AffineDialect",
        "memref::MemRefDialect",
        "arith::ArithmeticDialect"
    ];
}

def AffineToLLVM : Pass<"affine-to-llvm", "ModuleOp"> {
    let summary = "Lower Affine to LLVM.";
    let description = [{
    Lower implementation in Affine dialect to LLVM IR.
    }];

    let constructor = "createAffineToLLVM()";
    let dependentDialects = [
        "func::FuncDialect",
        "relay::RelayDialect",
        "AffineDialect",
        "memref::MemRefDialect",
        "arith::ArithmeticDialect",
        "LLVM::LLVMDialect"
    ];
}

#endif // TVM_MLIR_CONVERSION_PASSES
