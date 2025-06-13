#!/usr/bin/env python3
"""
画两张核心图：① 全局折线 (Entropy / KL / Top1) ② 位置-熵热图
"""
import json, os, argparse, matplotlib.pyplot as plt, seaborn as sns, numpy as np
plt.rcParams['figure.dpi'] = 150

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric_dir', default='analysis/metrics')
    ap.add_argument('--fig_dir',   default='figures')
    args = ap.parse_args(); os.makedirs(args.fig_dir, exist_ok=True)

    glob = json.load(open(f"{args.metric_dir}/distribution_metrics.json"))
    steps = sorted(map(int, glob))
    ent  = [glob[str(s)]['entropy_mean']            for s in steps]
    kl   = [glob[str(s)]['kl_from_uniform_mean']    for s in steps]
    top1 = [glob[str(s)]['top1_prob_mean']          for s in steps]

    # 折线
    plt.figure(figsize=(7,4))
    plt.plot(steps, ent,  'b-o', label='Entropy')
    plt.plot(steps, kl,   'r-s', label='KL (uniform)')
    plt.twinx(); plt.plot(steps, top1,'g-^', label='Top-1 prob')
    plt.axvline(100000, ls='--', c='k', alpha=.4)
    plt.title("Global output metrics"); plt.xlabel("Iteration")
    plt.legend(loc='lower center'); plt.tight_layout()
    plt.savefig(f"{args.fig_dir}/global_metric_lines.png")
    plt.close()

    # 热图
    pos_js = json.load(open(f"{args.metric_dir}/position_sensitivity_results.json"))
    pos_ids = sorted(int(k.split('_')[1]) for k in pos_js)
    mat = np.array([[pos_js[f'position_{p}']['entropy'][str(s)]
                     for s in steps] for p in pos_ids])
    plt.figure(figsize=(7,3))
    sns.heatmap(mat, yticklabels=pos_ids, xticklabels=steps, cmap='viridis_r')
    plt.xlabel("Iteration"); plt.ylabel("Position")
    plt.title("Position-wise entropy"); plt.tight_layout()
    plt.savefig(f"{args.fig_dir}/entropy_heatmap.png")
    plt.close()
    print("==> figures saved to", args.fig_dir)

if __name__ == "__main__":
    main()