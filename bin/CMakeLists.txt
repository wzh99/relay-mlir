add_llvm_executable(test test.cpp)
target_link_libraries(test MLIRRelay MLIRFuncDialect)
