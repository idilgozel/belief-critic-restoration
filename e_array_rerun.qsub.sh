#!/bin/bash -l
# Round-2 SGE array job for the GLEAM rerun package on UCL Myriad.
# Submit from ~/Scratch/gleam (11 configs x 5 seeds = 55 tasks):
#   mkdir -p logs
#   qsub -N gleam_rerun -t 1-55 -v CFG=configs_rerun.txt e_array_rerun.qsub.sh
#
# Task id maps over configs_rerun.txt x seeds {1..5}:
#   IDX  = (SGE_TASK_ID-1) % NCFG + 1   (config line)
#   SEED = (SGE_TASK_ID-1) / NCFG + 1   (seed 1..5)
#
# WALL TIME: GLEAM rows (1-2,5-11) are ~2 h each (round-1 wall_s ~6700 s per
# 10M-step run). The two MuJoCo skip4 rows (3-4) step the simulator 4x per
# policy step, so at the default 10M total-timesteps they run ~4x longer than a
# non-skip MuJoCo run and may exceed 36 h. If they time out, either raise h_rt
# for just those task ids or add `--total-timesteps 2500000` to lines 3-4 to
# hold simulator-steps (sample budget) constant with the non-skip controls.
#$ -l h_rt=36:0:0
#$ -l mem=4G
#$ -pe smp 4
#$ -cwd
#$ -j y
#$ -o logs/

set -e
module purge
module load default-modules 2>/dev/null || true
module load python3/3.11 2>/dev/null || module load python3/recommended
source ~/Scratch/gleam/venv/bin/activate
export OMP_NUM_THREADS=4
mkdir -p logs

CFG=${CFG:-configs_rerun.txt}
# guard: a missing CFG in $PWD would otherwise fail every array task silently at
# the wc/sed below. No effect when the file is present.
[[ -f "$CFG" ]] || { echo "CFG '$CFG' not found in $PWD" >&2; exit 1; }
NCFG=$(wc -l < "$CFG")
IDX=$(( (SGE_TASK_ID - 1) % NCFG + 1 ))
SEED=$(( (SGE_TASK_ID - 1) / NCFG + 1 ))
ARGS=$(sed -n "${IDX}p" "$CFG")

echo "task $SGE_TASK_ID -> config line $IDX seed $SEED: $ARGS"
python ppo_torch.py $ARGS --seed $SEED --out results_${JOB_NAME}.jsonl
