/*
 * Comm-touching PETSc/SLEPc constructors. Rust never sees MPI_Comm.
 * Typed EPSSet*/STSet* live in src/slepc_eps.rs. No *SetFromOptions.
 */
#include <petscmat.h>
#include <slepceps.h>
#include <stdint.h>

PetscErrorCode rgmin_slepc_initialized(PetscBool *flag)
{
  return PetscInitialized(flag);
}

PetscErrorCode rgmin_slepc_eps_create(EPS *eps)
{
  return EPSCreate(PETSC_COMM_SELF, eps);
}

PetscErrorCode rgmin_slepc_mat_create_shell(int64_t n, void *ctx, Mat *A)
{
  PetscInt N = (PetscInt)n;
  return MatCreateShell(PETSC_COMM_SELF, N, N, N, N, ctx, A);
}
