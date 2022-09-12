#include <iostream>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "tvm-mlir/Dialect/Relay/RelayDialect.h"
#include "tvm-mlir/Dialect/Relay/RelayOps.h"

using namespace mlir;

int main(int argc, char const *argv[]) {
    MLIRContext ctx;
    ctx.loadDialect<relay::RelayDialect, func::FuncDialect>();

    OpBuilder builder(&ctx);
    auto mod = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(mod.getBody());

    auto tensorType = RankedTensorType::get({2, 3}, builder.getF32Type());
    llvm::ArrayRef<float> data{1., 2., 3., 4., 5., 6.};

    auto func = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), builder.getStringAttr("main"),
        builder.getFunctionType({}, tensorType));

    auto entry = func.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    auto value = DenseElementsAttr::get(tensorType, data);
    auto constOp = builder.create<relay::ConstantOp>(builder.getUnknownLoc(), value);
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), constOp.getResult());

    mod.dump();
}
