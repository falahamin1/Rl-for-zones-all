#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=72-155%20
#SBATCH --job-name=zone-encoders-sf
#SBATCH --output=zone-encoders-sf.%A_%a.out
#SBATCH --qos=cpu-long

# Extension of submit_vrep_gnn.sbatch (--array=48-71%12, job-name=
# zone-encoders-vg) for the gridtimed seed extension (/tmp/SWEEP_FINAL.md
# Task C) -- array indices 72-155 in the SAME run_one.TARGET_RUNS list,
# appended, not a separate manifest. gridtimed/T2 and T3 showed per-seed
# bimodal outcomes for deepset_v/gnn at 3 seeds (some seeds solve the
# instance, some never rise off the initial return); this extends
# gridtimed to 10 seeds (0-2 existing + 3-9 new) across ALL SIX encoders,
# not just the two bimodal ones, so the comparison stays on equal footing.
# modesel is untouched -- its variance was already tight at 3 seeds.
#
# %20 throttle: raised from the original %12 since this is now a larger
# batch (84 vs 24/48 tasks) -- a starting value, not tuned against
# Alpine's actual acpu queue depth at submission time. Raise/lower by
# resubmitting with --array=72-155%<N> to override.
#
# QOS is cpu-long, NOT long -- acpu only accepts admin, cpu-normal,
# cpu-long (per /tmp/ENCODERS_SETUP.md). The older nav-sweep-*-gapfill.slurm
# scripts still say --qos=long, which acpu will reject; not fixed here
# (out of this task's scope), flagged in the report instead.

module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

# pyudbm's native extensions were compiled against a newer libstdc++ (via
# the gcc/14.2.0 module) than what's installed system-wide on at least
# some acpu nodes -- conda's own libstdcxx-ng package (pulled in as a
# pyudbm build dependency) provides a compatible one, but nothing points
# the runtime linker at it unless we do so explicitly. Without this,
# `import pyudbm` fails with "GLIBCXX_3.4.32 not found" on affected nodes.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# WORK_DIR: set to wherever this repo lives on Alpine -- same convention
# as submit.sbatch / submit_vrep_gnn.sbatch.
WORK_DIR="/projects/amfa5003/RL-for-zones"
cd "$WORK_DIR" || exit 1

# Idempotent/resumable: run_one.py checks results/ckpt/{run_id}.pt on
# start and continues from there automatically -- resubmitting this exact
# script (same array range) is always safe, no flag needed. Same
# checkpoint/resume contract, same atomic writes, same jsonl
# byte-offset truncation on resume as the original 72 runs -- run_one.py
# itself is unchanged in this regard, only TARGET_RUNS grew.
python3 run_one.py
