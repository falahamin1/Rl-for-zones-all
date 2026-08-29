#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-47%12
#SBATCH --job-name=zone-encoders
#SBATCH --output=zone-encoders.%A_%a.out
#SBATCH --qos=cpu-long

# %12 above throttles to 12 concurrent array tasks -- a starting value,
# not tuned against Alpine's actual acpu queue depth. Raise/lower by
# resubmitting with --array=0-47%<N> to override.
#
# QOS is cpu-long, NOT long -- acpu only accepts admin, cpu-normal,
# cpu-long (per /tmp/ENCODERS_SETUP.md). The older nav-sweep-*-gapfill.slurm
# scripts still say --qos=long, which acpu will reject; not fixed here
# (out of this task's scope), flagged in the report instead.

module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

# WORK_DIR: set this to wherever this repo lives on Alpine before
# submitting (same convention as b3-extend.slurm) -- e.g.:
#   WORK_DIR=/projects/$USER/RL-for-zones
WORK_DIR="$HOME/RL-for-zones"
cd "$WORK_DIR" || exit 1

# Idempotent/resumable: run_one.py checks results/ckpt/{run_id}.pt on
# start and continues from there automatically -- resubmitting this exact
# script (same array range) is always safe, no flag needed.
python3 run_one.py
