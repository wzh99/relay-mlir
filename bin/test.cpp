#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "tvm-mlir/Dialect/Relay/RelayDialect.hpp"
#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
#include "tvm-mlir/Frontend/RelayImporter.hpp"
#include "tvm/ir/module.h"
#include "tvm/parser/parser.h"
#include "tvm/relay/expr.h"

auto code = R"d(#[version = "0.0.5"]
def @main(%x: Tensor[(1, 2, 3), float32]) {
  meta[relay.Constant][0]
}

#[metadata]
{
  "root": 1, 
  "nodes": [
    {
      "type_key": ""
    }, 
    {
      "type_key": "Map", 
      "keys": [
        "relay.Constant"
      ], 
      "data": [2]
    }, 
    {
      "type_key": "Array", 
      "data": [3]
    }, 
    {
      "type_key": "relay.Constant", 
      "attrs": {
        "_checked_type_": "0", 
        "data": "0", 
        "span": "0"
      }
    }
  ], 
  "b64ndarrays": [
    "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAgAAAAIgAQACAAAAAAAAAAMAAAAAAAAAGAAAAAAAAAAAAIA/AAAAQAAAQEAAAIBAAACgQAAAwEA="
  ], 
  "attrs": {"tvm_version": "0.8.0"}
})d";

using namespace mlir;

int main(int argc, char const *argv[]) {
    MLIRContext ctx;
    ctx.loadDialect<relay::RelayDialect, func::FuncDialect>();

    // OpBuilder builder(&ctx);
    // auto mod = ModuleOp::create(builder.getUnknownLoc());
    // builder.setInsertionPointToEnd(mod.getBody());

    // auto tensorType = RankedTensorType::get({2, 3}, builder.getF32Type());
    // std::vector<float> data{1., 2., 3., 4., 5., 6.};

    // auto func = builder.create<func::FuncOp>(
    //     builder.getUnknownLoc(), builder.getStringAttr("main"),
    //     builder.getFunctionType({}, tensorType));

    // auto entry = func.addEntryBlock();
    // builder.setInsertionPointToEnd(entry);
    // auto value = DenseElementsAttr::get<float>(tensorType, data);
    // auto constOp =
    //     builder.create<relay::ConstantOp>(builder.getUnknownLoc(), value);
    // builder.create<func::ReturnOp>(builder.getUnknownLoc(),
    //                                constOp.getResult());

    // mod.dump();

    auto irmod = tvm::IRModule::FromText(code, "from_string");
    // std::cout << tvm::AsText(irmod);
    relay::ImportRelay(irmod, "from_string", ctx);
}
