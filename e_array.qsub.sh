set -e
module purge
module load default-modules 2>/dev/null || true
module load python3/3.11 2>/dev/null || module load python3/recommended
source ~/Scratch/gleam/venv/bin/activate
export OMP_NUM_THREADS=4
mkdir -p logs

NCFG=$(wc -l < "$CFG")
IDX=$(( (SGE_TASK_ID - 1) % NCFG + 1 ))
SEED=$(( (SGE_TASK_ID - 1) / NCFG + 1 ))
ARGS=$(sed -n "${IDX}p" "$CFG")

echo "task $SGE_TASK_ID -> config line $IDX seed $SEED: $ARGS"
python ppo_torch.py $ARGS --seed $SEED --out results_${JOB_NAME}.jsonl
