#!/usr/bin/env bash
# 一键跑完 0→2 步。
# 用法: bash analysis/run_analysis.sh out/simple_graph_1_1_120_100
set -e
CKPT_DIR=$1
STEPS="0 5000 10000 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000"

python analysis/0_extract_logits.py \
       --ckpt_dir "$CKPT_DIR" \
       --steps $STEPS \
       --n_tokens 2000 \
       --outfile analysis/logits.npz

python analysis/1_build_metrics.py \
       --logits_npz analysis/logits.npz \
       --outdir analysis/metrics

python analysis/2_visualize_outputs.py \
       --metric_dir analysis/metrics \
       --fig_dir figures

echo "===== 全部完成！图表在 figures/ ，JSON 在 analysis/metrics/ ====="