#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "tvm-mlir/Conversion/Passes.hpp"
#include "tvm-mlir/Dialect/Relay/Passes.hpp"
#include "tvm-mlir/Dialect/Relay/RelayDialect.hpp"
#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
#include "tvm-mlir/Frontend/RelayImporter.hpp"
#include "tvm-mlir/Support/Common.hpp"
#include "tvm-mlir/Support/MemRef.hpp"
#include "tvm-mlir/Transforms/Passes.hpp"
#include "tvm/ir/module.h"

using namespace mlir;
namespace cl = llvm::cl;
namespace path = llvm::sys::path;
namespace fs = llvm::sys::fs;

static auto inputPath = cl::opt<std::string>(cl::Positional);
static auto outputDir = cl::opt<std::string>(cl::Positional);

static void populatePasses(PassManager &pm) {
    pm.addPass(relay::createShapeInference());
    pm.addPass(relay::createOpFusion());
    pm.addPass(createRelayToAffine());
    pm.addPass(createOptimizeAffine());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createAffineToLLVM());
}

static bool shouldPrint(Pass *pass, Operation *op) {
    return isa<ModuleOp>(op) || cast<func::FuncOp>(op).getSymName() == "main";
}

int main(int argc, char const *argv[]) {
    // Parse command line arguments
    cl::ParseCommandLineOptions(argc, argv, "TVM-MLIR Sample Program");

    // Initialize MLIR context
    MLIRContext mlirCtx;
    mlirCtx
        .loadDialect<relay::RelayDialect, func::FuncDialect, scf::SCFDialect>();
    mlirCtx.disableMultithreading();

    // Configure pass manager
    PassManager pm(&mlirCtx, PassManager::Nesting::Implicit);
    populatePasses(pm);

    // Parse Relay source
    auto fileOrErr = llvm::MemoryBuffer::getFile(inputPath, true);
    if (auto err = fileOrErr.getError())
        Fatal("Cannot open file {}: {}", inputPath, err.message());
    auto buffer = fileOrErr->get()->getBuffer().str();
    auto irmod = tvm::IRModule::FromText(buffer, inputPath);

    // Create file to dump IR
    auto inputFilename = path::filename(inputPath);
    SmallVector<char> outFilename(inputFilename.begin(), inputFilename.end());
    path::replace_extension(outFilename, "mlir");
    SmallVector<char> outPathBuf(outputDir.begin(), outputDir.end());
    path::append(outPathBuf, outFilename);
    StringRef outputPath(outPathBuf.data(), outPathBuf.size());
    std::error_code err;
    llvm::raw_fd_ostream outStream(outputPath, err);
    if (err) Fatal("Cannot write to file {}: {}", outputPath, err.message());
    pm.enableIRPrinting([](Pass *, Operation *) { return false; }, shouldPrint,
                        true, false, false, outStream);

    // Import and compile
    auto mlirMod = relay::ImportRelay(irmod, inputPath, mlirCtx);
    if (pm.run(mlirMod).failed()) Fatal("Failed to run passes");

    // Export to LLVM
    registerLLVMDialectTranslation(mlirCtx);
    llvm::LLVMContext llvmCtx;
    auto llvmMod = translateModuleToLLVMIR(mlirMod, llvmCtx, inputPath);
    if (!llvmMod) Fatal("Failed to emit LLVM IR");
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    auto optPpl = makeOptimizingTransformer(3, 0, nullptr);
    if (auto err = optPpl(llvmMod.get())) Fatal("Failed to optimize LLVM IR");

    // Write LLVM IR to file
    path::replace_extension(outPathBuf, "ll");
    {
        StringRef outputPath(outPathBuf.data(), outPathBuf.size());
        std::error_code err;
        llvm::raw_fd_ostream outStream(outputPath, err);
        if (err)
            Fatal("Cannot write to file {}: {}", outputPath, err.message());
        llvmMod->print(outStream, nullptr);
    }

    return 0;
}
