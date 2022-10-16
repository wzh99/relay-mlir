#include "relay-mlir/Frontend/RelayImporter.hpp"

#include "OpConverter.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "relay-mlir/Dialect/Relay/RelayOps.hpp"
#include "relay-mlir/Support/Common.hpp"
#include "tvm/relay/expr_functor.h"

namespace mlir {
namespace relay {

class RelayImporter
    : private tvm::relay::ExprFunctor<Value(const tvm::relay::Expr &)> {
public:
    RelayImporter(const std::string &srcPath, MLIRContext &ctx)
        : srcPath(srcPath), builder(&ctx) {}

    ModuleOp Import(tvm::IRModule tvmMod);

private:
    using Base = tvm::relay::ExprFunctor<Value(const tvm::relay::Expr &)>;

    std::string srcPath;
    OpBuilder builder;
    std::unordered_map<tvm::relay::Expr, Value, tvm::ObjectHash,
                       tvm::ObjectEqual>
        exprValueMap;

    Value VisitExpr(const tvm::relay::Expr &n) override;
    Value VisitExpr_(const tvm::relay::ConstantNode *constant) override;
    Value VisitExpr_(const tvm::relay::VarNode *var) override;
    Value VisitExpr_(const tvm::relay::CallNode *call) override;

    Location cvtLoc(const tvm::Span &span) {
        return FileLineColLoc::get(builder.getStringAttr(srcPath), span->line,
                                   span->column);
    }
};

ModuleOp ImportRelay(tvm::IRModule mod, const std::string &srcPath,
                     MLIRContext &ctx) {
    return RelayImporter(srcPath, ctx).Import(mod);
}

inline static llvm::ArrayRef<int64_t> cvtTVMShape(
    const tvm::runtime::Array<tvm::PrimExpr> &relayShape) {
    std::vector<int64_t> shape;
    for (const auto &dim : relayShape) {
        auto imm = dim.as<tvm::IntImmNode>();
        if (!imm) Fatal("Shape dimension is not constant.");
        shape.push_back(imm->value);
    }
    return llvm::makeArrayRef(shape);
}

static Type getF32Type(OpBuilder &b) { return b.getF32Type(); }

static std::unordered_map<tvm::DataType, Type (*)(OpBuilder &)> typeMap{
    {tvm::DataType::Float(32), getF32Type}};

inline static Type cvtTVMDataType(const tvm::DataType &dtype,
                                  OpBuilder &builder) {
    if (typeMap.count(dtype))
        return typeMap[dtype](builder);
    else
        Fatal("Data type is not supported.");
}

inline static TensorType cvtRelayTensorType(
    const tvm::relay::TensorTypeNode *type, OpBuilder &builder) {
    auto shape = cvtTVMShape(type->shape);
    auto dtype = cvtTVMDataType(type->dtype, builder);
    return RankedTensorType::get(shape, dtype);
}

inline static TensorType extractRelayVarType(const tvm::relay::Var &var,
                                             OpBuilder &builder) {
    auto &type = var->type_annotation;
    if (!type.defined())
        Fatal("Relay variable {} is not type-annotated.",
              var->name_hint().c_str());
    auto tvmTensorType = type.as<tvm::relay::TensorTypeNode>();
    if (!tvmTensorType)
        Fatal("Variable {} is not of tensor type.", var->name_hint().c_str());
    return cvtRelayTensorType(tvmTensorType, builder);
}

ModuleOp RelayImporter::Import(tvm::IRModule tvmMod) {
    // Create MLIR module
    auto mod = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(mod.getBody());

    // Create prototype of main function
    auto relayMain = tvmMod->functions.at(tvmMod->GetGlobalVar("main"))
                         .as<tvm::relay::FunctionNode>();
    std::vector<mlir::Type> paramTypes;
    for (const auto &var : relayMain->params)
        paramTypes.push_back(extractRelayVarType(var, builder));
    auto funcType = builder.getFunctionType(paramTypes, llvm::None);
    auto mainFunc =
        builder.create<func::FuncOp>(cvtLoc(relayMain->span), "main", funcType);

    // Add parameter values to symbol table
    auto entry = mainFunc.addEntryBlock();
    for (auto i = 0u; i < entry->getNumArguments(); i++) {
        auto &var = relayMain->params[i];
        auto value = entry->getArgument(i);
        exprValueMap.insert({var, value});
    }

    // Insert operations to function body
    builder.setInsertionPointToStart(entry);
    auto ret = VisitExpr(relayMain->body);
    builder.create<func::ReturnOp>(cvtLoc(relayMain->body->span), ret);

    return mod;
}

Value RelayImporter::VisitExpr(const tvm::relay::Expr &expr) {
    if (exprValueMap.count(expr)) return exprValueMap[expr];
    auto ret = Base::VisitExpr(expr);
    exprValueMap.insert({expr, ret});
    return ret;
}

template <class T>
static DenseElementsAttr createDense(RankedTensorType type, char *data,
                                     size_t size) {
    return DenseElementsAttr::get(
        type, llvm::makeArrayRef(reinterpret_cast<T *>(data),
                                 reinterpret_cast<T *>(data + size)));
}

static std::unordered_map<tvm::DataType, DenseElementsAttr (*)(RankedTensorType,
                                                               char *, size_t)>
    denseCreateFn{{tvm::DataType::Float(32), createDense<float>}};

Value RelayImporter::VisitExpr_(const tvm::relay::ConstantNode *constant) {
    // Get tensor type for this constant
    auto tensor = constant->data;
    auto shape =
        llvm::makeArrayRef(tensor->shape, tensor->shape + tensor->ndim);
    auto tvmDType = tensor.DataType();
    auto elemType = cvtTVMDataType(tvmDType, builder);
    auto type = RankedTensorType::get(shape, elemType);
    auto size = tvm::runtime::GetDataSize(*tensor.operator->());

    // Create constant operation
    if (!denseCreateFn.count(tensor.DataType()))
        Fatal("Data type is not supported.");
    auto attr = denseCreateFn[tvmDType](
        type, reinterpret_cast<char *>(tensor->data), size);
    auto op = builder.create<ConstantOp>(cvtLoc(constant->span), type, attr);

    return op.getResult();
}

Value RelayImporter::VisitExpr_(const tvm::relay::VarNode *var) {
    return exprValueMap.at(tvm::GetRef<tvm::relay::Var>(var));
}

Value RelayImporter::VisitExpr_(const tvm::relay::CallNode *call) {
    auto relayOp = call->op.as<tvm::relay::OpNode>();
    if (!relayOp) Fatal("Call to non-operator expression is not supported.");
    std::vector<Value> operands;
    for (auto &arg : call->args) operands.push_back(VisitExpr(arg));
    auto op = ConvertRelayOp(relayOp->name, operands, call->attrs,
                             cvtLoc(call->span), builder);
    return op->getResult(0);
}

}  // namespace relay
}  // namespace mlir
