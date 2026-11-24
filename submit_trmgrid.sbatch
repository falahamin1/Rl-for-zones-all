#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=168-287%20
#SBATCH --job-name=zone-encoders-trmgrid
#SBATCH --output=zone-encoders-trmgrid.%A_%a.out
#SBATCH --qos=cpu-long

# trmgrid extension (/tmp/TRMGRID_SWEEP.md) -- the branching-RM domain.
# Array indices 168-287 in the SAME run_one.TARGET_RUNS list, appended,
# not a separate manifest: 2 TRMGRID_INSTANCES (G_SMALL, G_LARGE) x 6
# encoders (mlp_point, deepset_h, deepset_v, cnn_dbm, cnn_proj, gnn) x 10
# TRMGRID_SEEDS = 120 tasks. Each trains 300k env steps, plateau early
# stopping OFF (uniform budget, no early stopping, same convention as
# every other block in this sweep).
#
# domain="trmgrid" is a brand-new run_id prefix, structurally distinct
# from every pre-existing "modesel__"/"gridtimed__" run_id -- verified
# explicitly (not just reasoned about) by
# test_run_one.py::test_trmgrid_run_ids_do_not_collide_with_anything.
#
# %20 caps concurrency at 20 (same style as the original sweeps); not
# tuned against Alpine's actual acpu queue depth at submission time.
#
# QOS is cpu-long, NOT long -- acpu only accepts admin, cpu-normal,
# cpu-long (per /tmp/ENCODERS_SETUP.md).

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
# as submit.sbatch / submit_vrep_gnn.sbatch / submit_sweep_final.sbatch /
# submit_shaping.sbatch.
WORK_DIR="/projects/amfa5003/RL-for-zones"
cd "$WORK_DIR" || exit 1

# Idempotent/resumable: run_one.py checks results/ckpt/{run_id}.pt on
# start and continues from there automatically -- resubmitting this exact
# script (same array range) is always safe, no flag needed. Plateau early
# stopping is OFF by default -- every one of these 120 runs trains to
# exactly 300k steps, uninterrupted, no flag needed either.
python3 run_one.py
