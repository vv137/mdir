// The dialects a valid MDIR program, including a staged model, may contain.
#ifndef MDIR_DIALECT_PROGRAMDIALECTS_H
#define MDIR_DIALECT_PROGRAMDIALECTS_H
namespace mlir {
class DialectRegistry;
}
namespace mdir {
/// Shared by construction/canonical printing and the lowering registry.
void insertProgramDialects(mlir::DialectRegistry &);
} // namespace mdir
#endif
