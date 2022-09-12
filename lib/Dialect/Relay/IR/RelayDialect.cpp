#include "tvm-mlir/Dialect/Relay/RelayDialect.h"

#include "tvm-mlir/Dialect/Relay/RelayOps.h"
#include "tvm-mlir/Dialect/Relay/RelayOpsDialect.cpp.inc"

namespace mlir {
namespace relay {

void RelayDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "tvm-mlir/Dialect/Relay/RelayOps.cpp.inc"
        >();
}

}  // namespace relay
}  // namespace mlir