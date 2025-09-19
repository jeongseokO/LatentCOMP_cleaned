#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --job-name=HeatMap
#SBATCH --mem=50G
#SBATCH --gres=gpu:A6000:1      # 평가에는 1개 GPU만 필요하다면 조정
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH -o slurm_output/%x_%j.out
#SBATCH -e slurm_output/%x_%j.err

# ===========================
# 0) 기본 환경
# ===========================
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HOME=/data2/jeongseokoh/hub
# conda 활성화
source /data2/${USER}/.bashrc
source /data2/jeongseokoh/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# HF 로그인
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
else
  echo "[Warning] HF_TOKEN not set. Push to Hub will fail if enabled."
fi


srun python attn_heatmap_tri.py \
  --modeling ./lopa_llama_modeling.py \
  --repo    jeongseokoh/LoPA_Llama3.1_8B_16_Lowers \
  --device   cuda \
  --attn_impl eager \
  --lower_k  8 \
  --document "Cottee's and Faygo both produce soft drinks." \
  --question "Cottee's and Faygo both produce what?" \
  --response "They both produce soft drinks." \
  --outdir   ./_attn_vis