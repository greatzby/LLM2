#!/usr/bin/env python3
"""
将 logits.npz => distribution_metrics.json / position_sensitivity_results.json / kl_jump.json
兼容 object-array 存储格式。
"""
import numpy as np, json, argparse, os, math

# ---------- helpers ----------
def entropy(arr):   # arr [..., V]
    return -np.sum(arr * np.log2(arr + 1e-12), axis=-1)

def gini(arr):
    return 1.0 - np.sum(arr ** 2, axis=-1)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logits_npz', required=True)
    ap.add_argument('--outdir', default='analysis/metrics')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data    = np.load(args.logits_npz, allow_pickle=True)
    probs_o = data['probs']    # object array len = n_ckpt
    tgt_o   = data['targets']
    steps   = data['steps']

    global_js, pos_js = {}, {}

    for idx, step in enumerate(steps):
        # ---- 转成 float32 ndarray, squeeze 掉长度为 1 的时间维 ----
        p = np.asarray(list(probs_o[idx]), dtype=np.float32) if probs_o.dtype==object else probs_o[idx]
        t = np.asarray(list(tgt_o[idx]),   dtype=np.int64 ) if tgt_o.dtype  ==object else tgt_o[idx]
        if p.ndim == 3 and p.shape[1] == 1:     # (N,1,V) -> (N,V)
            p = p.squeeze(1); t = t.squeeze(1)

        # --------- 全局指标 ---------
        ent_mean = float(entropy(p).mean())
        top1     = float(p.max(-1).mean())
        top5     = float(np.sort(p, axis=-1)[:, -5:].sum(-1).mean())
        kl_uni   = float((-entropy(p) + math.log2(p.shape[-1])).mean())
        eff      = float((1 / (p ** 2).sum(-1)).mean())
        g        = float(gini(p).mean())
        m2s      = float((p.max(-1) /
                          (np.partition(p, -2, axis=-1)[:, -2] + 1e-12)).mean())

        global_js[int(step)] = dict(
            entropy_mean=ent_mean,
            top1_prob_mean=top1,
            top5_prob_mean=top5,
            kl_from_uniform_mean=kl_uni,
            effective_choices_mean=eff,
            gini_mean=g,
            max_to_second_ratio_mean=m2s
        )

        # --------- position 级 ---------
        ent_pos = entropy(p).mean(0)                    # [T-1]
        acc_pos = (p.argmax(-1) == t).mean(0)           # [T-1]
        for pos_id, (e, a) in enumerate(zip(ent_pos, acc_pos)):
            d = pos_js.setdefault(f"position_{pos_id+3}", dict(entropy={}, accuracy={}))
            d["entropy"][int(step)]  = float(e)
            d["accuracy"][int(step)] = float(a)

    # --------- ΔKL( t｜t-1 ) ---------
    step_sorted = sorted(global_js)
    kl_vals = [global_js[s]['kl_from_uniform_mean'] for s in map(str, step_sorted)]
    kl_jump = {step_sorted[i]: float(kl_vals[i] - kl_vals[i-1])
               for i in range(1, len(step_sorted))}

    json.dump(global_js, open(f"{args.outdir}/distribution_metrics.json", 'w'), indent=2)
    json.dump(pos_js,    open(f"{args.outdir}/position_sensitivity_results.json", 'w'), indent=2)
    json.dump(kl_jump,   open(f"{args.outdir}/kl_jump.json", 'w'), indent=2)
    print("==> metrics saved to", args.outdir)

if __name__ == "__main__":
    main()