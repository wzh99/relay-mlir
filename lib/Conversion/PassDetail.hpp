#pragma once

#include "mlir/IR/BuiltinDialect.h"
#include "relay-mlir/Conversion/Passes.hpp"

namespace mlir {

class ModuleOp;

#define GEN_PASS_CLASSES
#include "relay-mlir/Conversion/Passes.h.inc"

}  // namespace mlir
