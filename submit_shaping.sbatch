#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=156-167%12
#SBATCH --job-name=zone-encoders-shape
#SBATCH --output=zone-encoders-shape.%A_%a.out
#SBATCH --qos=cpu-long

# Shaping experiment (/tmp/PART1_FINAL.md item 3) -- the deferred-credit-
# assignment mechanism test. Array indices 156-167 in the SAME
# run_one.TARGET_RUNS list, appended, not a separate manifest: 2
# SHAPING_ENCODERS (mlp_point, deepset_h) x 3 SEEDS x 2 variants
# (unshaped control, shaped) = 12 tasks, gridtimed/T2 only.
#
# The shaped variant adds a potential-based reward
# (gridtimed.gym_env.GridTimedTrainEnv's switch_shaping_reward=10.0, ~10%
# of REWARD_SUCCESS=100.0) the first time each switch is activated per
# episode. The unshaped control is a FRESH, uninterrupted 300k run under
# this same code (not a reuse of the original sweep's gridtimed/T2/
# mlp_point or deepset_h runs, which went through item 1's
# early-stop-then-resume path -- a different training trajectory shape
# than a clean run, and not a fair comparison partner for the shaped arm).
#
# Only 12 tasks -- %12 lets every task in this batch run concurrently
# given no other constraint; not tuned against Alpine's actual acpu queue
# depth at submission time.
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
# as submit.sbatch / submit_vrep_gnn.sbatch / submit_sweep_final.sbatch.
WORK_DIR="/projects/amfa5003/RL-for-zones"
cd "$WORK_DIR" || exit 1

# Idempotent/resumable: run_one.py checks results/ckpt/{run_id}.pt on
# start and continues from there automatically -- resubmitting this exact
# script (same array range) is always safe, no flag needed. Plateau early
# stopping is OFF by default (item 1) -- every one of these 12 runs trains
# to exactly 300k steps, uninterrupted, no flag needed either.
python3 run_one.py
