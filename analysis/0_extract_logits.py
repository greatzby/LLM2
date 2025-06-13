#!/usr/bin/env python3
"""
从若干 checkpoint (文件名形如 5000_ckpt_20.pt) 推理验证集，保存 softmax 概率。
"""
import torch, argparse, os, math, numpy as np, pickle, importlib, tqdm

# --------------------------------------------------
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = ckpt['model_args']
    GPT = importlib.import_module('model').GPT
    GPTConfig = importlib.import_module('model').GPTConfig
    model = GPT(GPTConfig(**model_args)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, model_args

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--steps', nargs='+', type=int, required=True)
    ap.add_argument('--n_tokens', type=int, default=2000,
                    help='验证集抽取 token 数 (>=1000 保证稳定)')
    ap.add_argument('--outfile', default='analysis/logits.npz')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------- 找 meta / val.bin ----------
    meta_path = None
    for root, _, files in os.walk('data'):
        if 'meta.pkl' in files:
            meta_path = os.path.join(root, 'meta.pkl'); break
    assert meta_path, "找不到 meta.pkl，请确认 data 目录结构"
    meta = pickle.load(open(meta_path, 'rb'))
    block_size = meta['block_size']
    val_bin = os.path.join(os.path.dirname(meta_path), 'val.bin')
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')
    assert len(val_data) > (block_size + 1) * args.n_tokens, "验证集太小，n_tokens 调小点"

    # ---------- 主循环 ----------
    probs_list, tgt_list, step_list = [], [], []
    for step in args.steps:
        ckpt_file = os.path.join(args.ckpt_dir, f'{step}_ckpt_20.pt')
        assert os.path.exists(ckpt_file), f"{ckpt_file} 不存在"
        print(f'>> loading {ckpt_file}')
        model, _ = load_model(ckpt_file, device)

        # 采样 n_tokens 序列，每序列长度 = block_size
        idx_seq = torch.randint(
            0, (len(val_data) - block_size - 1) // (block_size + 1),
            (args.n_tokens,)
        ) * (block_size + 1)
        all_tokens = []
        for pos in idx_seq:
            all_tokens.extend(range(pos, pos + block_size))
        all_tokens = torch.tensor(all_tokens, dtype=torch.long, device=device)
        seq = torch.from_numpy(val_data[all_tokens.cpu().numpy()].astype(np.int64)).to(device)
        seq = seq.view(args.n_tokens, block_size)  # [N, T]

        with torch.no_grad():
            # logits for first T-1 positions => probability分布 shape (N, T-1, V)
            logits, _ = model(seq[:, :-1], targets=None)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        probs_list.append(probs.astype(np.float16))
        tgt_list.append(seq[:, 1:].cpu().numpy())
        step_list.append(step)
        print(f'   collected probs shape {probs.shape}')

    np.savez_compressed(
        args.outfile,
        probs=np.array(probs_list, dtype=object),
        targets=np.array(tgt_list, dtype=object),
        steps=np.array(step_list)
    )
    print("==> logits saved to", args.outfile)

if __name__ == "__main__":
    main()