#include "tvm-mlir/Dialect/Relay/RelayOps.h"

#include "mlir/IR/OpImplementation.h"
#include "tvm-mlir/Dialect/Relay/RelayDialect.h"

#define GET_OP_CLASSES
#include "tvm-mlir/Dialect/Relay/RelayOps.cpp.inc"

namespace mlir {
namespace relay {

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       DenseElementsAttr value) {
    build(odsBuilder, odsState, value.getType(), value);
}

LogicalResult ConstantOp::inferReturnTypeComponents(
    MLIRContext *, llvm::Optional<Location>, ValueShapeRange, DictionaryAttr,
    RegionRange, llvm::SmallVectorImpl<ShapedTypeComponents> &) {
    return success();
}

}  // namespace relay
}  // namespace mlir
