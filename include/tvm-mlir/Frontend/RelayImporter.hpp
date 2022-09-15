#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "tvm/ir/module.h"

namespace mlir {
namespace relay {

ModuleOp ImportRelay(tvm::IRModule mod, const std::string &srcPath, MLIRContext &ctx);

}  // namespace relay
}  // namespace mlir

