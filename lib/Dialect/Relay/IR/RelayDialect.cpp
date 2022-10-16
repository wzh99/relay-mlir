#include "relay-mlir/Dialect/Relay/RelayDialect.hpp"

#include "relay-mlir/Dialect/Relay/RelayOps.hpp"
#include "relay-mlir/Dialect/Relay/RelayOpsDialect.cpp.inc"

namespace mlir {
namespace relay {

void RelayDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "relay-mlir/Dialect/Relay/RelayOps.cpp.inc"
        >();
}

}  // namespace relay
}  // namespace mlir