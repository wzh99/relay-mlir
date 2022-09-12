#ifndef MLIR_TVM_RELAY_OPS
#define MLIR_TVM_RELAY_OPS

include "tvm-mlir/Dialect/Relay/RelayDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Relay_Op<string mnemonic, list<Trait> traits> : Op<Relay_Dialect, mnemonic, traits> {}

// Constant
def Relay_ConstantOp : Relay_Op<"constant", [ConstantLike, NoSideEffect, InferTensorType]> {
  let summary = "Constant operator";
  let description = [{
    Represents a constant value.
  }];

  let arguments = (ins ElementsAttr:$value);

  let results = (outs AnyStaticShapeTensor:$output);

  let builders = [OpBuilder<(ins "::mlir::ElementsAttr":$value)>];

//   let hasCustomAssemblyFormat = 1;
}

#endif
