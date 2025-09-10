#!/usr/bin/env python
import argparse, os, re, csv, torch, random
from typing import List, Tuple
from utils import SimpleVocab, ParallelTextDataset, collate_pad, PAD_ID, BOS_ID, EOS_ID, corpus_bleu, SentencePieceVocab
from model import Transformer

# tqdm is optional; fall back gracefully if missing
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---------- helpers ----------
def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def split_indices_perm(n: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    gen = torch.Generator(); gen.manual_seed(seed)
    idx = torch.randperm(n, generator=gen).tolist()
    n_val = int(n * val_ratio); n_test = int(n * test_ratio)
    return idx[n_val+n_test:], idx[:n_val], idx[n_val:n_val+n_test]

def split_indices_head(n: int, val_ratio: float, test_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    n_val = int(n * val_ratio); n_test = int(n * test_ratio)
    return list(range(n_val+n_test, n)), list(range(0, n_val)), list(range(n_val, n_val+n_test))

def vocab_from_tokens(tokens: List[str]) -> SimpleVocab:
    return SimpleVocab(tokens)

def load_vocabs_from_ckpt(ckpt):
    tok = ckpt.get("tokenizer", "basic")
    if tok == "spm" and ("src_spm_proto" in ckpt and "tgt_spm_proto" in ckpt):
        src_proto = ckpt["src_spm_proto"]
        tgt_proto = ckpt["tgt_spm_proto"]
        # Allow either serialized bytes or file paths
        if isinstance(src_proto, (bytes, bytearray)):
            src_vocab = SentencePieceVocab(model_proto=src_proto)
        else:
            src_vocab = SentencePieceVocab(model_file=str(src_proto))
        if isinstance(tgt_proto, (bytes, bytearray)):
            tgt_vocab = SentencePieceVocab(model_proto=tgt_proto)
        else:
            tgt_vocab = SentencePieceVocab(model_file=str(tgt_proto))
    else:
        src_vocab = vocab_from_tokens(ckpt["src_vocab"]) if "src_vocab" in ckpt else None
        tgt_vocab = vocab_from_tokens(ckpt["tgt_vocab"]) if "tgt_vocab" in ckpt else None
        if src_vocab is None or tgt_vocab is None:
            raise RuntimeError("Checkpoint missing vocabulary information. Re-train or include SPM protos.")
    return src_vocab, tgt_vocab

@torch.no_grad()
def greedy_decode(model, src_ids, max_len=64):
    device = next(model.parameters()).device
    ys = torch.full((src_ids.size(0), 1), BOS_ID, dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model(src_ids, ys)
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
        ys = torch.cat([ys, next_id], dim=1)
        if (next_id == EOS_ID).all(): break
    return ys

@torch.no_grad()
def beam_search(model, src_ids, beam_size=5, max_len=64, alpha=0.7):
    # expects batch size 1 for simplicity
    device = next(model.parameters()).device
    assert src_ids.size(0) == 1, "beam_search here assumes batch size = 1"
    beams = [(torch.tensor([[BOS_ID]], device=device, dtype=torch.long), 0.0)]
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == EOS_ID:
                new_beams.append((seq, score)); continue
            logits = model(src_ids, seq)          # (1,T,V)
            logp = torch.log_softmax(logits[:, -1, :], dim=-1)
            vals, idxs = torch.topk(logp, beam_size, dim=-1)
            for k in range(beam_size):
                tok = idxs[0, k].view(1, 1)
                new_seq = torch.cat([seq, tok], dim=1)
                new_score = score + float(vals[0, k])
                new_beams.append((new_seq, new_score))
        def norm(item):
            seq, sc = item; L = max(1, seq.size(1)); return sc / (L ** alpha)
        beams = sorted(new_beams, key=norm, reverse=True)[:beam_size]
        if all(seq[0, -1].item() == EOS_ID for seq, _ in beams): break
    best_seq, _ = max(beams, key=lambda x: x[1] / (max(1, x[0].size(1)) ** alpha))
    return best_seq

@torch.no_grad()
def topk_sampling(model, src_ids, k=50, max_len=64, temperature=1.0):
    device = next(model.parameters()).device
    ys = torch.full((src_ids.size(0), 1), BOS_ID, dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model(src_ids, ys)
        logits = logits[:, -1, :] / max(1e-6, temperature)
        logp = torch.log_softmax(logits, dim=-1)
        vals, idxs = torch.topk(logp, k, dim=-1)
        choice = torch.distributions.Categorical(logits=vals).sample()
        next_id = idxs.gather(-1, choice.unsqueeze(-1))
        ys = torch.cat([ys, next_id], dim=1)
        if (next_id.squeeze(-1) == EOS_ID).all(): break
    return ys

def decode_sentence(model, src_vocab, tgt_vocab, sent: str, strategy: str,
                    beam_size: int, alpha: float, topk: int, temperature: float, max_len: int, device: str):
    if hasattr(src_vocab, "is_spm") and getattr(src_vocab, "is_spm"):
        s_ids = src_vocab.encode_sentence(sent.strip(), add_bos_eos=True, max_len=max_len)
    else:
        s_ids = [src_vocab.stoi.get(tok, 0) for tok in ["<bos>"] + sent.strip().split() + ["<eos>"]]
    src_tensor = torch.tensor(s_ids, dtype=torch.long, device=device).unsqueeze(0)
    if strategy == "greedy":
        out = greedy_decode(model, src_tensor, max_len=max_len)
    elif strategy == "beam":
        out = beam_search(model, src_tensor, beam_size=beam_size, max_len=max_len, alpha=alpha)
    else:
        out = topk_sampling(model, src_tensor, k=topk, max_len=max_len, temperature=temperature)
    return tgt_vocab.decode(out[0].tolist())  # list[str], stops at EOS

def pick_indices(n: int, k: int, mode: str, seed: int, indices_str: str = None) -> List[int]:
    if k <= 0: return []
    if mode == "indices":
        assert indices_str, "--sample-indices must be provided when --sample-mode indices"
        idxs = [int(x) for x in indices_str.split(",")]
        for x in idxs:
            if not (0 <= x < n):
                raise ValueError(f"Sample index {x} out of range 0..{n-1}")
        return idxs[:k]
    if mode == "first":
        return list(range(min(k, n)))
    # random
    rnd = random.Random(seed)
    return rnd.sample(range(n), k if k <= n else n)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True, help="Path to specific checkpoint file (.pt)")
    ap.add_argument("--data-src", required=True)
    ap.add_argument("--data-tgt", required=True)
    ap.add_argument("--split-mode", choices=["perm", "head"], default="perm",
                    help="Use the same split recipe you trained with")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # strategy params
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-len", type=int, default=None, help="Override decode length; else use ckpt arg or 96")
    ap.add_argument("--max-examples", type=int, default=None, help="Evaluate only the first N test examples")
    ap.add_argument("--progress", action="store_true", help="Show tqdm progress bar during decoding")
    ap.add_argument("--out-csv", type=str, default=None)
    # Optional overrides to force SentencePiece usage if ckpt lacks protos
    ap.add_argument("--src-spm-proto", type=str, default=None,
                    help="Path to source SentencePiece model proto (.model). Forces tokenizer=spm")
    ap.add_argument("--tgt-spm-proto", type=str, default=None,
                    help="Path to target SentencePiece model proto (.model). Forces tokenizer=spm")
    # NEW: sampling controls
    ap.add_argument("--samples", type=int, default=3, help="Number of samples to print/save")
    ap.add_argument("--sample-mode", choices=["random","first","indices"], default="random")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--sample-indices", type=str, default=None,
                    help="Comma-separated 0-based indices into the test set (used when --sample-mode indices)")
    args = ap.parse_args()

    # evaluate the specified checkpoint
    if not os.path.isfile(args.ckpt_path):
        raise SystemExit(f"Checkpoint file not found: {args.ckpt_path}")
    ckpts = [args.ckpt_path]

    # load data + build test split once (per ckpt we only reuse indices)
    src_all = load_lines(args.data_src)
    tgt_all = load_lines(args.data_tgt)
    assert len(src_all) == len(tgt_all), "Source/target size mismatch"
    n = len(src_all)

    all_rows = []
    sample_rows = []  # NEW

    for path in ckpts:
        base = os.path.basename(path)
        print(f"[info] evaluating {path}")
        ckpt = torch.load(path, map_location=args.device)
        print(f"[info] loaded model from {path}")

        ck_args = ckpt.get("args", {})

        # If SPM protos are provided via CLI, override tokenizer info
        if args.src_spm_proto and args.tgt_spm_proto:
            ckpt = dict(ckpt)  # shallow copy to allow injection
            ckpt["tokenizer"] = "spm"
            ckpt["src_spm_proto"] = args.src_spm_proto
            ckpt["tgt_spm_proto"] = args.tgt_spm_proto
            print("[info] using SentencePiece models from CLI overrides")

        # split params from ckpt (fallback to common defaults)
        val_ratio = float(ck_args.get("val_ratio", 0.02))
        test_ratio= float(ck_args.get("test_ratio", 0.02))
        seed      = int(ck_args.get("seed", 42))

        if args.split_mode == "perm":
            _, _, test_idx = split_indices_perm(n, val_ratio, test_ratio, seed)
        else:
            _, _, test_idx = split_indices_head(n, val_ratio, test_ratio)

        src_test_all = [src_all[i] for i in test_idx]
        tgt_test_all = [tgt_all[i] for i in test_idx]

        # optionally limit evaluation size
        if args.max_examples is not None:
            src_test_all = src_test_all[:args.max_examples]
            tgt_test_all = tgt_test_all[:args.max_examples]

        # --- pick sample indices (within the *test* subset) ---
        k = args.samples
        sample_idx_local = pick_indices(len(src_test_all), k, args.sample_mode, args.sample_seed, args.sample_indices)
        print(f"[samples] picking indices from test set (size={len(src_test_all)}): {sample_idx_local}")

        # vocab + model
        src_vocab, tgt_vocab = load_vocabs_from_ckpt(ckpt)

        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=ck_args.get("d_model", 320),
            num_layers=ck_args.get("layers", 5),
            num_heads=ck_args.get("heads", 5),
            d_ff=ck_args.get("d_ff", 1280),
            dropout=ck_args.get("dropout", 0.1),
            posenc=ck_args.get("posenc", "rope"),
        ).to(args.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        max_len = args.max_len or int(ck_args.get("max_len", 96))

        # --- decode ONLY the selected samples and print/save them ---
        printed = []
        for i in sample_idx_local:
            s = src_test_all[i]
            t = tgt_test_all[i].strip().split()
            g = decode_sentence(model, src_vocab, tgt_vocab, s, "greedy",
                                beam_size=args.beam_size, alpha=args.alpha,
                                topk=args.topk, temperature=args.temperature,
                                max_len=max_len, device=args.device)
            b = decode_sentence(model, src_vocab, tgt_vocab, s, "beam",
                                beam_size=args.beam_size, alpha=args.alpha,
                                topk=args.topk, temperature=args.temperature,
                                max_len=max_len, device=args.device)
            tp = decode_sentence(model, src_vocab, tgt_vocab, s, "topk",
                                 beam_size=args.beam_size, alpha=args.alpha,
                                 topk=args.topk, temperature=args.temperature,
                                 max_len=max_len, device=args.device)

            row = {
                "checkpoint": base,
                "idx": i,
                "src": s,
                "ref": " ".join(t),
                "greedy": " ".join(g),
                "beam": " ".join(b),
                "topk": " ".join(tp),
            }
            sample_rows.append(row)
            printed.append(row)

        # pretty print to stdout
        print("\n=== Sample outputs (input / greedy / beam / top-k / reference) ===")
        for r in printed:
            print(f"\n[idx {r['idx']}]")
            print(f"  SRC: {r['src']}")
            print(f"  GRD: {r['greedy']}")
            print(f"  BEM: {r['beam']}")
            print(f"  TPK: {r['topk']}")
            print(f"  REF: {r['ref']}")

    # write samples CSV next to checkpoint (unless overridden)
    if args.out_csv:
        out_csv = args.out_csv
    else:
        checkpoint_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]
        checkpoint_dir = os.path.dirname(args.ckpt_path)
        out_csv = os.path.join(checkpoint_dir, f"{checkpoint_name}_samples.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint","idx","src","ref","greedy","beam","topk"])
        w.writeheader(); w.writerows(sample_rows)

    print(f"\nSaved samples CSV → {out_csv}")
    print("Done.")

if __name__ == "__main__":
    main()