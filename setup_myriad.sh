
set -e
module purge
module load default-modules 2>/dev/null || true
module load python3/3.11 2>/dev/null || module load python3/recommended

python3 -m venv ~/Scratch/gleam/venv
source ~/Scratch/gleam/venv/bin/activate
pip install --upgrade pip
# CPU torch is sufficient (64-dim MLPs); swap for cuda wheel if using -l gpu=1.
# Versions pinned for round-2 reproducibility (validated locally on 2026-07-03).
# NOTE: round-1 (2026-07-02) ran UNPINNED "latest"; confirm these match the
# round-1 Myriad venv with `pip freeze` and adjust if they differ.
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
pip install "gymnasium[mujoco]==1.3.0" mujoco==3.10.0 numpy==2.4.6 scipy==1.17.1

# smoke test (~30 s)
python ppo_torch.py --env-id GLEAMBench-v0 --total-timesteps 20000 \
    --exp-name smoke --out smoke.jsonl
echo "SETUP OK — smoke result:"
cat smoke.jsonl
