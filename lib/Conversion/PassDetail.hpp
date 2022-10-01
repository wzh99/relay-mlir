#pragma once

#include "mlir/IR/BuiltinDialect.h"
#include "tvm-mlir/Conversion/Passes.hpp"
#include "tvm-mlir/Dialect/Relay/RelayDialect.hpp"

namespace mlir {

class AffineDialect;

namespace func {
class FuncDialect;
}

namespace memref {
class MemRefDialect;
}

namespace arith {
class ArithmeticDialect;
}

namespace LLVM {
class LLVMDialect;
}

class ModuleOp;

#define GEN_PASS_CLASSES
#include "tvm-mlir/Conversion/Passes.h.inc"

}  // namespace mlir
