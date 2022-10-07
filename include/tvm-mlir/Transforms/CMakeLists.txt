set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TVMMLIRTransforms)
add_public_tablegen_target(TVMMLIRTransformsPassIncGen)
add_mlir_doc(Passes TransformsPasses ./ -gen-pass-doc)
