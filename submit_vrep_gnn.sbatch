#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=48-71%12
#SBATCH --job-name=zone-encoders-vg
#SBATCH --output=zone-encoders-vg.%A_%a.out
#SBATCH --qos=cpu-long

# Extension of submit.sbatch (--array=0-47%12, job-name=zone-encoders) for
# the 2 new encoders (deepset_v, gnn) added in /tmp/VREP_FEASIBILITY.md --
# array indices 48-71 in the SAME run_one.TARGET_RUNS list, appended, not
# a separate manifest. Different job-name/output pattern than the
# original purely so squeue/ls distinguish the two submissions; run_one.py
# itself doesn't care which sbatch file launched it.
#
# %12 above throttles to 12 concurrent array tasks -- same starting value
# as the original submission, not independently re-tuned. Raise/lower by
# resubmitting with --array=48-71%<N> to override.
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
# as submit.sbatch.
WORK_DIR="/projects/amfa5003/RL-for-zones"
cd "$WORK_DIR" || exit 1

# Idempotent/resumable: run_one.py checks results/ckpt/{run_id}.pt on
# start and continues from there automatically -- resubmitting this exact
# script (same array range) is always safe, no flag needed. Same
# checkpoint/resume contract, same atomic writes, same jsonl
# byte-offset truncation on resume as the original 48 runs -- run_one.py
# itself is unchanged in this regard, only TARGET_RUNS grew.
python3 run_one.py
