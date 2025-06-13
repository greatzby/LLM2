#!/usr/bin/env python3
"""
把 logits.npz -> three json files；兼容 object array / 任意形状。
"""
import numpy as np, json, argparse, os, math

def entropy(arr):  # [..., V]
    return -np.sum(arr * np.log2(arr + 1e-12), axis=-1)

def gini(arr):
    return 1 - np.sum(arr ** 2, axis=-1)

def to_ndarray(x):
    """object array -> float32 ndarray"""
    if isinstance(x, np.ndarray) and x.dtype != object:
        return x
    return np.array(list(x), dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logits_npz', required=True)
    ap.add_argument('--outdir', default='analysis/metrics')
    args = ap.parse_args(); os.makedirs(args.outdir, exist_ok=True)

    pack   = np.load(args.logits_npz, allow_pickle=True)
    probsO, tgtO, steps = pack['probs'], pack['targets'], pack['steps']

    global_js, pos_js = {}, {}

    for i, step in enumerate(steps):
        p = to_ndarray(probsO[i])   # (N,T?,V)
        t = to_ndarray(tgtO[i])     # (N,T?)
        # 若 p 为 (N,1,V) 则 squeeze 掉第 1 维
        if p.ndim == 3 and p.shape[1] == 1:
            p = p[:, 0, :]
            t = t[:, 0]
        # 现在形状 (N,T,V)
        if p.ndim == 2:  # T==1 的特殊情况
            p = p[:, None, :]
            t = t[:, None]

        # -------- 全局指标 --------
        ent_mean = float(entropy(p).mean())
        top1     = float(p.max(-1).mean())
        top5     = float(np.sort(p, axis=-1)[:, :, -5:].sum(-1).mean())
        kl_uni   = float((-entropy(p) + math.log2(p.shape[-1])).mean())
        eff      = float((1 / (p ** 2).sum(-1)).mean())
        g        = float(gini(p).mean())
        m2s      = float((p.max(-1) /
                          (np.partition(p, -2, axis=-1)[:, :, -2] + 1e-12)).mean())

        global_js[int(step)] = dict(
            entropy_mean=ent_mean,
            top1_prob_mean=top1,
            top5_prob_mean=top5,
            kl_from_uniform_mean=kl_uni,
            effective_choices_mean=eff,
            gini_mean=g,
            max_to_second_ratio_mean=m2s
        )

        # -------- 位置级 --------
        ent_pos = entropy(p).mean(0)                # (T,)
        acc_pos = (p.argmax(-1) == t).mean(0)       # (T,)
        for j, (e, a) in enumerate(zip(ent_pos, acc_pos)):
            d = pos_js.setdefault(f"position_{j+3}", dict(entropy={}, accuracy={}))
            d["entropy"][int(step)]  = float(e)
            d["accuracy"][int(step)] = float(a)

    # -------- ΔKL --------
    st_sorted = sorted(global_js)
    kl_vals   = [global_js[s]['kl_from_uniform_mean'] for s in st_sorted]
    kl_jump   = {st_sorted[i]: float(kl_vals[i]-kl_vals[i-1])
                 for i in range(1,len(st_sorted))}

    json.dump(global_js, open(f"{args.outdir}/distribution_metrics.json", 'w'), indent=2)
    json.dump(pos_js,    open(f"{args.outdir}/position_sensitivity_results.json", 'w'), indent=2)
    json.dump(kl_jump,   open(f"{args.outdir}/kl_jump.json", 'w'), indent=2)
    print("metrics saved to", args.outdir)

if __name__ == "__main__":
    main()