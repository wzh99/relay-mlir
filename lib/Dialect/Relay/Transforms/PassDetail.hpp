#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "tvm-mlir/Dialect/Relay/RelayDialect.hpp"

namespace mlir {
namespace relay {

#define GEN_PASS_CLASSES
#include "tvm-mlir/Dialect/Relay/Passes.h.inc"

}  // namespace relay

}  // namespace mlir
