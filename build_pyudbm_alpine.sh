#!/usr/bin/env bash
# Build pyudbm's native extensions (UUtils/UDBM/UCDD/UTAP + Python
# bindings) from source on Alpine, WITHOUT root.
#
# Run this INTERACTIVELY on a LOGIN node, before submitting the SLURM
# array -- it needs outbound internet (CMake FetchContent pulls libxml2
# from GitHub; conda pulls packages from conda-forge) and compute nodes on
# most clusters don't have that. Confirms pyudbm actually imports on Linux
# before burning queue time on 48 array tasks that would otherwise all die
# in `import zoneviews` -> `import pyudbm`.
#
# Mirrors pyudbm/cibw/linux_before_build.sh (the script pyudbm's own CI
# uses to build official Linux wheels), with one change: that script calls
# `apt-get`/`yum install m4` if m4 is missing, which needs root you won't
# have on a shared cluster. Everything here goes through conda instead
# (into the same amfa-custom-env submit.sbatch activates), which also
# means the autotools-build-bison/flex-from-scratch fallback in the
# original script isn't needed either -- conda-forge ships prebuilt
# bison/flex/cmake/ninja/libxml2 packages directly.
#
# Usage: ./build_pyudbm_alpine.sh [path-to-repo-root]
#   (defaults to $HOME/RL-for-zones)
set -euo pipefail

REPO_ROOT="$(python3 -c "import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())" "${1:-$HOME/RL-for-zones}")"
PYUDBM_DIR="$REPO_ROOT/pyudbm"
if [[ ! -d "$PYUDBM_DIR" ]]; then
    echo "error: $PYUDBM_DIR not found -- pass the repo root as the first argument" >&2
    exit 1
fi
cd "$PYUDBM_DIR"

# ---- -1. Rewrite SSH-style GitHub URLs to HTTPS ----
# UTAP's CMake FetchContent pulls doctest (and would pull libxml2, though
# conda's libxml2 preempts that) via `git@github.com:...` -- SSH clone
# URLs, which need a GitHub SSH key set up on THIS machine. That key lives
# on your Mac, not Alpine, so the clone fails with "Permission denied
# (publickey)". Rewriting to https:// sidesteps needing any auth at all
# for a public repo. Same fix pyudbm/cibw/linux_before_build.sh uses.
git config --global --add url."https://github.com/".insteadOf git@github.com:
git config --global --add url."https://github.com/".insteadOf ssh://git@github.com/

# ---- 0. Conda environment (same one submit.sbatch activates) ----
module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

# conda's own compiler packages (gcc_linux-64/gxx_linux-64) set these
# automatically via activation hooks; the cluster's `module load gcc` has
# no idea this conda env exists, so headers/libs conda installed (icu,
# libxml2, ...) don't get found even though they're right there --
# e.g. libxml2's encoding.h #includes <unicode/ucnv.h> (from conda's icu
# package), findable only if $CONDA_PREFIX/include is on the search path.
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- 1. Compiler ----
# Prefer a cluster-provided module compiler over conda's cross-compiler
# packages -- it's simpler and Alpine almost certainly has one. The exact
# module name varies by cluster; `gcc` is the common default. If this
# fails, check `module avail gcc` and edit the name below, or fall through
# to the conda-forge compilers.
if module load gcc 2>/dev/null; then
    echo "Using cluster module: gcc"
    CC="$(command -v gcc)"
    CXX="$(command -v g++)"
else
    echo "No 'gcc' module found -- falling back to conda-forge compilers." >&2
    conda install -y -c conda-forge gcc_linux-64 gxx_linux-64
    # These packages install conda activate.d hooks that set $CC/$CXX
    # themselves on (re)activation -- don't hand-guess the binary name.
    conda deactivate
    conda activate amfa-custom-env
    if [[ -z "${CC:-}" || -z "${CXX:-}" ]]; then
        echo "error: conda-forge compiler activation didn't set \$CC/\$CXX -- inspect 'conda list gcc_linux-64' and this env's activate.d scripts manually." >&2
        exit 1
    fi
fi
echo "CC=$CC"
echo "CXX=$CXX"
export CC CXX

# ---- 2. Build prerequisites, all via conda (no root, no system package manager) ----
# cmake<4 matches pyudbm/requirements-build.txt's own pin: CMake 4.x hard-
# errors on any project (including UUtils's auto-fetched `doctest` test
# dependency) that declares cmake_minimum_required() below 3.5, where
# CMake 3.x would just warn. Letting conda pick the latest cmake (4.x)
# breaks the UUtils build the moment it FetchContents doctest.
conda install -y -c conda-forge "cmake<4" ninja pybind11 bison flex m4 libxml2

export CMAKE_GENERATOR=Ninja
BUILD_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
FLEX_EXECUTABLE="$(command -v flex)"
BISON_EXECUTABLE="$(command -v bison)"
BIN_INSTALL="$(python3 -c "import pathlib; print(pathlib.Path('bin_install').resolve())")"

resolve_glob() {
    python3 -c "
import glob, pathlib, sys
matches = sorted(glob.glob(sys.argv[1]))
if not matches:
    raise SystemExit(f'No path matched: {sys.argv[1]}')
print(pathlib.Path(matches[0]).resolve())
" "$1"
}

# ---- 3. Phase 1: build + install UUtils -> UDBM -> UCDD -> UTAP ----
# Same dependency order pyudbm/cibw/linux_before_build.sh uses -- each
# stage installs into $BIN_INSTALL so the next stage's find_package(...
# CONFIG REQUIRED) can locate it.
echo "=== Building UUtils ==="
cmake -G "$CMAKE_GENERATOR" -S UUtils -B UUtils_build \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build UUtils_build --config Release -- -j"$BUILD_JOBS"
cmake --install UUtils_build --prefix "$BIN_INSTALL" --config Release
UUTILS_DIR="$(resolve_glob "$BIN_INSTALL/lib*/cmake/UUtils")"

echo "=== Building UDBM ==="
cmake -G "$CMAKE_GENERATOR" -S UDBM -B UDBM_build \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_BUILD_TYPE=Release -DUUtils_DIR="$UUTILS_DIR"
cmake --build UDBM_build --config Release -- -j"$BUILD_JOBS"
cmake --install UDBM_build --prefix "$BIN_INSTALL" --config Release
UDBM_DIR="$(resolve_glob "$BIN_INSTALL/lib*/cmake/UDBM")"

echo "=== Building UCDD ==="
cmake -G "$CMAKE_GENERATOR" -S UCDD -B UCDD_build \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$BIN_INSTALL" \
    -DUUtils_DIR="$UUTILS_DIR" -DUDBM_DIR="$UDBM_DIR" -DFIND_FATAL=OFF
cmake --build UCDD_build --config Release -- -j"$BUILD_JOBS"
cmake --install UCDD_build --prefix "$BIN_INSTALL" --config Release

echo "=== Building UTAP ==="
# UTAP's own cmake/libxml2.cmake will use the conda-forge libxml2 we just
# installed (found via find_package) rather than fetching+building its
# own copy from GitHub -- faster and doesn't need FetchContent's network
# access for this specific dependency (bison/flex are still needed above,
# already covered by conda).
cmake -G "$CMAKE_GENERATOR" -S UTAP -B UTAP_build \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$BIN_INSTALL" \
    -DFLEX_EXECUTABLE="$FLEX_EXECUTABLE" -DBISON_EXECUTABLE="$BISON_EXECUTABLE"
cmake --build UTAP_build --config Release -- -j"$BUILD_JOBS"
cmake --install UTAP_build --prefix "$BIN_INSTALL" --config Release

# ---- 4. Phase 2: build the actual Python bindings ----
# The top-level pyudbm/CMakeLists.txt find_package()s all four libraries
# above via $BINSTALL_DIR (falls back to ./bin_install if unset -- setting
# it explicitly here for clarity, matching what we just installed to).
export BINSTALL_DIR="$BIN_INSTALL"
pip install -U ninja "pybind11[global]" "cmake<4"
pip install -e "$PYUDBM_DIR"

# ---- 5. Smoke test ----
echo "=== Smoke test ==="
python3 -c "
import pyudbm
print('pyudbm imported OK, version', pyudbm.__version__)
ctx = pyudbm.Context(['x', 'y'])
print('Context construction OK -- build looks good.')
"
