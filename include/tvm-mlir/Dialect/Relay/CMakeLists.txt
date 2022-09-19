add_mlir_dialect(RelayOps relay)
add_mlir_doc(RelayDialect RelayDialect Relay/ -gen-dialect-doc)
add_mlir_doc(RelayOps RelayOps Relay/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name RelayOpt)
add_public_tablegen_target(MLIRRelayPassIncGen)
add_mlir_doc(Passes RelayPasses ./ -gen-pass-doc)
