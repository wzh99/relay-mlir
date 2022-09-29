#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "tvm-mlir/Conversion/RelayToAffine.hpp"
#include "tvm-mlir/Dialect/Relay/Passes.hpp"
#include "tvm-mlir/Dialect/Relay/RelayDialect.hpp"
#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
#include "tvm-mlir/Frontend/RelayImporter.hpp"
#include "tvm-mlir/Support/Common.hpp"
#include "tvm/ir/module.h"
#include "tvm/parser/parser.h"
#include "tvm/relay/expr.h"

auto code = R"d(#[version = "0.0.5"]
def @main(%x: Tensor[(1, 2), float32], %w1: Tensor[(2, 2), float32], %b1: Tensor[(2), float32],
    %w2: Tensor[(4, 2), float32], %b2: Tensor[(4), float32]) {
    %0 = nn.dense(%x, %w1, units=None);
    %1 = nn.bias_add(%0, %b1, axis=1);
    %2 = nn.relu(%1);
    %3 = nn.dense(%2, %w2, units=None);
    %4 = nn.bias_add(%3, %b2, axis=1);
    %4
})d";

using namespace mlir;

int main(int argc, char const *argv[]) {
    MLIRContext ctx;
    ctx.loadDialect<relay::RelayDialect, func::FuncDialect>();
    auto irmod = tvm::IRModule::FromText(code, "from_string");
    auto mod = relay::ImportRelay(irmod, "from_string", ctx);
    PassManager pm(&ctx, PassManager::Nesting::Implicit);
    pm.addPass(relay::createShapeInference());
    pm.addPass(relay::createOpFusion());
    pm.addPass(createRelayToAffine());
    pm.run(mod).succeeded();
    mod.dump();
}
