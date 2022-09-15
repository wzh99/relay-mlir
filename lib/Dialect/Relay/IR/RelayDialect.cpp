#include "tvm-mlir/Dialect/Relay/RelayDialect.hpp"

#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
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