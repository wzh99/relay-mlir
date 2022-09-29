add_llvm_executable(test test.cpp)
target_link_libraries(test 
    PUBLIC 
    MLIRRelayDialect 
    MLIRFuncDialect 
    MLIRRelayFrontend 
    MLIRRelayTransforms
    MLIRRelayToAffine
)
