#!/usr/bin/env python3
"""
从若干 checkpoint (文件名形如 5000_ckpt_20.pt) 推理验证集，保存 softmax 概率。
"""
# ==== 新增：把项目根目录加入 sys.path =========================================
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# =============================================================================

import torch, argparse, os, math, numpy as np, pickle, importlib, random

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = ckpt['model_args']
    GPT       = importlib.import_module('model').GPT
    GPTConfig = importlib.import_module('model').GPTConfig
    model = GPT(GPTConfig(**model_args)).to(device)
    model.load_state_dict(ckpt['model']); model.eval()
    return model, model_args
# -------------------------- 其余内容保持不变 -------------------------------
# (从 “def main():” 开始的部分完全复用之前给你的版本)
import math
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--steps', nargs='+', type=int, required=True)
    ap.add_argument('--n_tokens', type=int, default=2000)
    ap.add_argument('--outfile', default='analysis/logits.npz')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 找 meta / val.bin ---
    meta_path = None
    for root, _, files in os.walk('data'):
        if 'meta.pkl' in files:
            meta_path = os.path.join(root, 'meta.pkl'); break
    if meta_path is None:
        raise RuntimeError("在 data/* 下找不到 meta.pkl")
    meta = pickle.load(open(meta_path, 'rb'))
    block_size = meta['block_size']
    val_bin = os.path.join(os.path.dirname(meta_path), 'val.bin')
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')

    max_tokens = (len(val_data) - block_size - 1) // (block_size + 1)
    if max_tokens <= 0:
        raise RuntimeError("验证集太小，无法抽取任何序列")
    if args.n_tokens > max_tokens:
        print(f"[WARN] 要求 {args.n_tokens} 个样本，但验证集只能提供 {max_tokens}，自动下调。")
        args.n_tokens = max_tokens

    probs_list, tgt_list, step_list = [], [], []
    for step in args.steps:
        ckpt = os.path.join(args.ckpt_dir, f'{step}_ckpt_20.pt')
        assert os.path.exists(ckpt), ckpt + " 不存在"
        model,_ = load_model(ckpt,device)

        idx = torch.randint(max_tokens,(args.n_tokens,)) * (block_size+1)
        gather = []
        for i in idx: gather.extend(range(i, i+block_size))
        seq = torch.from_numpy(val_data[np.array(gather)]).long().to(device)
        seq = seq.view(args.n_tokens, block_size)

        with torch.no_grad():
            logits,_ = model(seq[:,:-1]); probs = torch.softmax(logits.float(), -1).cpu().numpy()
        probs_list.append(probs.astype(np.float16))
        tgt_list.append(seq[:,1:].cpu().numpy())
        step_list.append(step)
        print(f"step {step} ✓  shape={probs.shape}")

    np.savez_compressed(args.outfile,
        probs=np.array(probs_list, dtype=object),
        targets=np.array(tgt_list, dtype=object),
        steps=np.array(step_list))
    print("saved to", args.outfile)

if __name__ == '__main__':
    main()