#!/usr/bin/env bash
# Build PRIMME, ChASE and libkrylov from source into PREFIX.
#
# Run inside the conda-forge backends environment: CONDA_PREFIX holds
# OpenBLAS and Open MPI, and mpicc / mpicxx are on PATH. GF12_PREFIX is
# a second conda prefix with gfortran 12 and OpenBLAS. libkrylov 1.1.0
# fails 146 of its 242 tests under gfortran 14 and 16 and passes all of
# them under 12, so this script runs that ctest before installing.
set -euo pipefail

prefix=${1:?usage: build_ci_backends.sh PREFIX [SRC]}
src=${2:-${PWD}/backend-src}
: "${CONDA_PREFIX:?run inside the backends environment}"
: "${GF12_PREFIX:?set GF12_PREFIX to the gfortran 12 prefix}"

PRIMME_REF=v3.2.3
CHASE_REF=v1.7.4
LIBKRYLOV_REF=v1.1.0
jobs=$(nproc)

mkdir -p "$prefix/lib" "$prefix/include" "$src"

git clone --quiet --depth 1 --branch "$PRIMME_REF" \
  https://github.com/primme/primme.git "$src/primme"
make -C "$src/primme" lib -j"$jobs" CC=gcc PYTHON=python3 CFLAGS="-O2 -fPIC -DNDEBUG"
cp "$src/primme/lib/libprimme.a" "$prefix/lib/"
cp "$src/primme"/include/primme*.h "$prefix/include/"

git clone --quiet --depth 1 --branch "$CHASE_REF" \
  https://github.com/ChASE-library/ChASE.git "$src/chase"
cmake -S "$src/chase" -B "$src/chase/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DBLA_VENDOR=OpenBLAS \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_Fortran_COMPILER=gfortran
cmake --build "$src/chase/build" -j"$jobs"
cmake --install "$src/chase/build"

git clone --quiet --depth 1 --branch "$LIBKRYLOV_REF" \
  https://gitlab.com/libkrylov/libkrylov-stable.git "$src/libkrylov"
"$GF12_PREFIX/bin/gfortran" --version | head -n 1
cmake -S "$src/libkrylov" -B "$src/libkrylov/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$GF12_PREFIX" \
  -DCMAKE_C_COMPILER="$GF12_PREFIX/bin/gcc" \
  -DCMAKE_Fortran_COMPILER="$GF12_PREFIX/bin/gfortran" \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build "$src/libkrylov/build" -j"$jobs"
(cd "$src/libkrylov/build" && LD_LIBRARY_PATH="$GF12_PREFIX/lib" ctest -j"$jobs" --output-on-failure)
cmake --install "$src/libkrylov/build" --prefix "$prefix"
