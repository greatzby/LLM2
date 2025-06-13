#!/usr/bin/env python3
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch, argparse, os, numpy as np, pickle, importlib

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    GPT       = importlib.import_module('model').GPT
    GPTConfig = importlib.import_module('model').GPTConfig
    model = GPT(GPTConfig(**ckpt['model_args'])).to(device)
    model.load_state_dict(ckpt['model']); model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--steps', nargs='+', type=int, required=True)
    ap.add_argument('--n_tokens', type=int, default=2000)
    ap.add_argument('--outfile', default='analysis/logits.npz')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- meta / val.bin ----
    meta_path = next(p for p,_,f in os.walk('data') if 'meta.pkl' in f)+"/meta.pkl"
    meta = pickle.load(open(meta_path,'rb'))
    block_size = meta['block_size']
    val_bin = os.path.join(os.path.dirname(meta_path), 'val.bin')
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')
    max_tokens = (len(val_data) - block_size - 1)//(block_size+1)
    if args.n_tokens > max_tokens:
        print(f"[WARN] 要求 {args.n_tokens} 个样本，但验证集只能提供 {max_tokens}，自动下调。")
        args.n_tokens = max_tokens

    probs_all, tgt_all, step_all = [], [], []
    for step in args.steps:
        ckpt_file = f"{args.ckpt_dir}/{step}_ckpt_20.pt"
        assert os.path.exists(ckpt_file), ckpt_file+"不存在"
        model = load_model(ckpt_file, device)

        # 随机采样 token 序列
        anchors = torch.randint(max_tokens, (args.n_tokens,)) * (block_size+1)
        gather  = [i+j for i in anchors for j in range(block_size)]
        gather_idx = np.array(gather, dtype=np.int64)            # ← 修改 1
        seq_np = val_data[gather_idx].astype(np.int64)           # ← 修改 2
        seq = torch.from_numpy(seq_np).to(device)                # ← 修改 3
        seq = seq.view(args.n_tokens, block_size)

        with torch.no_grad():
            logits,_ = model(seq[:,:-1]); probs = torch.softmax(logits.float(), -1).cpu().numpy()

        probs_all.append(probs.astype(np.float16))
        tgt_all.append(seq[:,1:].cpu().numpy())
        step_all.append(step)
        print(f"step {step} ✓  probs {probs.shape}")

    np.savez_compressed(args.outfile,
        probs=np.array(probs_all, dtype=object),
        targets=np.array(tgt_all, dtype=object),
        steps=np.array(step_all))
    print("saved to", args.outfile)

if __name__ == '__main__':
    main()