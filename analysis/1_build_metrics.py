#!/usr/bin/env python3
"""
把 logits.npz → distribution_metrics.json / position_sensitivity_results.json / kl_jump.json
"""
import numpy as np, json, argparse, os, math, tqdm

def entropy(p): return -np.sum(p * np.log2(p + 1e-12), axis=-1)
def gini(p):    return 1 - np.sum(p ** 2, axis=-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logits_npz', required=True)
    ap.add_argument('--outdir', default='analysis/metrics')
    args = ap.parse_args(); os.makedirs(args.outdir, exist_ok=True)

    data = np.load(args.logits_npz, allow_pickle=True)
    probs, targets, steps = data['probs'], data['targets'], data['steps']

    global_js, pos_js = {}, {}
    for i, step in enumerate(steps):
        p = probs[i]    # (N, T-1, V)
        t = targets[i]  # (N, T-1)

        ent   = float(entropy(p).mean())
        top1  = float(p.max(-1).mean())
        top5  = float(np.sort(p, axis=-1)[:, :, -5:].sum(-1).mean())
        klu   = float((-entropy(p) + math.log2(p.shape[-1])).mean())
        eff   = float((1 / (p ** 2).sum(-1)).mean())
        gmean = float(gini(p).mean())
        m2s   = float((p.max(-1) /
                       (np.partition(p, -2, axis=-1)[:, :, -2] + 1e-12)).mean())

        global_js[int(step)] = dict(
            entropy_mean=ent,
            top1_prob_mean=top1,
            top5_prob_mean=top5,
            kl_from_uniform_mean=klu,
            effective_choices_mean=eff,
            gini_mean=gmean,
            max_to_second_ratio_mean=m2s,
        )

        ent_pos = entropy(p).mean(0)
        acc_pos = (p.argmax(-1) == t).mean(0)
        for j, (e, a) in enumerate(zip(ent_pos, acc_pos)):
            d = pos_js.setdefault(f"position_{j+3}", dict(entropy={}, accuracy={}))
            d["entropy"][int(step)] = float(e)
            d["accuracy"][int(step)] = float(a)

    # ΔKL
    step_sorted = sorted(global_js)
    kl_vals = [global_js[s]['kl_from_uniform_mean'] for s in step_sorted]
    kl_jump = {step_sorted[i]: float(kl_vals[i] - kl_vals[i - 1])
               for i in range(1, len(step_sorted))}

    json.dump(global_js, open(f"{args.outdir}/distribution_metrics.json", 'w'), indent=2)
    json.dump(pos_js,    open(f"{args.outdir}/position_sensitivity_results.json", 'w'), indent=2)
    json.dump(kl_jump,   open(f"{args.outdir}/kl_jump.json", 'w'), indent=2)
    print("==> metrics saved to", args.outdir)

if __name__ == "__main__":
    main()