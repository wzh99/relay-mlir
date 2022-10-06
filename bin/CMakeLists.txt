add_llvm_executable(test test.cpp)
target_link_libraries(test 
    PUBLIC 
    MLIRRelayDialect 
    MLIRRelayFrontend 
    MLIRRelayTransforms
    TVMMLIRConversion
    TVMMLIRTransforms
    MLIRExecutionEngine
    LLVMX86CodeGen
)
