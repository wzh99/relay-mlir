#include "tvm-mlir/Frontend/RelayImporter.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
#include "tvm-mlir/Support/Error.hpp"
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
    std::string srcPath;
    OpBuilder builder;
    std::unordered_map<tvm::relay::Expr, Value, tvm::ObjectHash,
                       tvm::ObjectEqual>
        exprValueMap;

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
        if (!imm) FatalError("Shape dimension is not constant.");
        shape.push_back(imm->value);
    }
    return llvm::makeArrayRef(shape);
}

static std::unordered_map<tvm::DataType, std::function<Type(OpBuilder &)>>
    typeMap{{tvm::DataType::Float(32),
             [](OpBuilder &b) { return b.getF32Type(); }}};

inline static Type cvtTVMDataType(const tvm::DataType &dtype,
                                  OpBuilder &builder) {
    if (typeMap.count(dtype))
        return typeMap[dtype](builder);
    else
        FatalError("Data type is not supported.");
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
        FatalError("Relay variable {} is not type-annotated.",
                   var->name_hint().c_str());
    auto tvmTensorType = type.as<tvm::relay::TensorTypeNode>();
    if (!tvmTensorType)
        FatalError("Variable {} is not of tensor type.",
                   var->name_hint().c_str());
    auto mlirTensorType = cvtRelayTensorType(tvmTensorType, builder);
    return cvtRelayTensorType(tvmTensorType, builder);
}

ModuleOp RelayImporter::Import(tvm::IRModule tvmMod) {
    // Create MLIR module
    auto mlirMod = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(mlirMod.getBody());

    // Create prototype of main function
    auto relayMain = tvmMod->functions.at(tvmMod->GetGlobalVar("main"))
                         .as<tvm::relay::FunctionNode>();
    std::vector<mlir::Type> paramTypes;
    for (const auto &var : relayMain->params)
        paramTypes.push_back(extractRelayVarType(var, builder));
    auto funcType = builder.getFunctionType(paramTypes, llvm::None);
    auto mainFunc = builder.create<func::FuncOp>(cvtLoc(relayMain->span),
                                                 "main", funcType);

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
    mlirMod.dump();

    return mlirMod;
}

template <class T>
static DenseElementsAttr createDense(RankedTensorType &&type, char *data,
                                     size_t size) {
    return DenseElementsAttr::get(
        std::move(type),
        llvm::makeArrayRef(reinterpret_cast<T *>(data),
                           reinterpret_cast<T *>(data + size)));
}

static std::unordered_map<
    tvm::DataType, DenseElementsAttr (*)(RankedTensorType &&, char *, size_t)>
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
        FatalError("Data type is not supported.");
    auto attr = denseCreateFn[tvmDType](
        std::move(type), reinterpret_cast<char *>(tensor->data), size);
    auto op = builder.create<ConstantOp>(cvtLoc(constant->span), attr);

    return op.getResult();
}

Value RelayImporter::VisitExpr_(const tvm::relay::VarNode *var) { return {}; }

Value RelayImporter::VisitExpr_(const tvm::relay::CallNode *call) { return {}; }

}  // namespace relay
}  // namespace mlir
