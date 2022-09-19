#pragma once

#include "mlir/IR/OpDefinition.h"
#include "tvm/ir/attrs.h"

namespace mlir {
namespace relay {

OpState ConvertRelayOp(const tvm::String &name, const std::vector<Value> &args,
                       const tvm::Attrs &attrs, Location loc,
                       OpBuilder &builder);

}
}  // namespace mlir
