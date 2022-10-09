#pragma once

#include "mlir/IR/BuiltinDialect.h"
#include "tvm-mlir/Conversion/Passes.hpp"

namespace mlir {

class ModuleOp;

#define GEN_PASS_CLASSES
#include "tvm-mlir/Conversion/Passes.h.inc"

}  // namespace mlir
