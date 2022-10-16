add_llvm_executable(sample sample.cpp)
target_link_libraries(sample 
    PUBLIC 
    MLIRRelayDialect 
    MLIRRelayFrontend 
    MLIRRelayTransforms
    RelayMLIRConversion
    RelayMLIRTransforms
    MLIRSCFToOpenMP
    MLIRExecutionEngine
    MLIROpenMPToLLVMIRTranslation
    LLVMX86CodeGen
)
