#!/usr/bin/env python3
import json, argparse, os, matplotlib.pyplot as plt, seaborn as sns, numpy as np
plt.rcParams['figure.dpi'] = 150

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric_dir', default='analysis/metrics')
    ap.add_argument('--fig_dir',   default='figures')
    args = ap.parse_args(); os.makedirs(args.fig_dir, exist_ok=True)

    glob = json.load(open(f"{args.metric_dir}/distribution_metrics.json"))
    steps = sorted(map(int, glob))
    ent  = [glob[str(s)]['entropy_mean']         for s in steps]
    kl   = [glob[str(s)]['kl_from_uniform_mean'] for s in steps]
    top1 = [glob[str(s)]['top1_prob_mean']       for s in steps]

    # 折线
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax2 = ax1.twinx()
    ax1.plot(steps, ent, 'b-o', label='Entropy')
    ax1.plot(steps, kl,  'r-s', label='KL(uniform)')
    ax2.plot(steps, top1,'g-^', label='Top1 prob')
    ax1.set_xlabel("Iteration"); ax1.set_title("Global output metrics")
    ax1.axvline(100000, ls='--', c='k', alpha=.4)
    ax1.legend(loc='upper left'); ax2.legend(loc='lower right')
    fig.tight_layout(); fig.savefig(f"{args.fig_dir}/global_metric_lines.png")

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
    print("figures saved to", args.fig_dir)

if __name__ == "__main__":
    main()