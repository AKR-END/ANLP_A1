# test.py
import argparse, torch
from utils import SimpleVocab, BOS_ID, EOS_ID
from model import Transformer

def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device)
    # The checkpoint stored vocab token lists; SimpleVocab expects a token list
    src_vocab, tgt_vocab = SimpleVocab(ckpt["src_vocab"]), SimpleVocab(ckpt["tgt_vocab"])
    args = ckpt["args"]
    model = Transformer(
        len(src_vocab), len(tgt_vocab),
        d_model=args["d_model"], num_layers=args["layers"],
        num_heads=args["heads"], d_ff=args["d_ff"],
        dropout=args["dropout"], posenc=args["posenc"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, src_vocab, tgt_vocab

def tokenize(vocab: SimpleVocab, s: str, max_len=128):
    toks = ["<bos>"] + s.strip().split()[:max_len-2] + ["<eos>"]
    ids = [vocab.stoi.get(t, 0) for t in toks]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

@torch.no_grad()
def greedy_decode(model, src, max_len=64):
    device = next(model.parameters()).device
    ys = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model(src, ys)                # (B, T, V)
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
        ys = torch.cat([ys, next_id], dim=1)   # append
        if (next_id == EOS_ID).all():
            break
    return ys

@torch.no_grad()
def beam_search(model, src, beam_size=4, max_len=64, alpha=0.7):
    """
    Classic length-normalized beam search (score / len^alpha).
    """
    device = next(model.parameters()).device
    B = src.size(0)
    assert B == 1, "Beam search implementation here assumes batch size = 1"
    beams = [(torch.full((1, 1), BOS_ID, dtype=torch.long, device=device), 0.0)]  # (seq, logprob)

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            # If already ended with EOS, keep as is (do not expand)
            if seq[0, -1].item() == EOS_ID:
                new_beams.append((seq, score))
                continue
            logits = model(src, seq)  # (1, T, V)
            logp = torch.log_softmax(logits[:, -1, :], dim=-1)  # (1, V)
            values, indices = torch.topk(logp, beam_size, dim=-1)  # (1, K)
            for k in range(beam_size):
                token = indices[0, k].view(1, 1)
                new_seq = torch.cat([seq, token], dim=1)
                new_score = score + float(values[0, k].item())
                new_beams.append((new_seq, new_score))

        # length-normalize and keep top beams
        def norm_score(item):
            seq, sc = item
            L = max(1, seq.size(1))   # length including BOS and generated tokens
            return sc / (L ** alpha)

        beams = sorted(new_beams, key=norm_score, reverse=True)[:beam_size]
        # Stop early if all beams ended
        if all(seq[0, -1].item() == EOS_ID for seq, _ in beams):
            break

    # Return the best beam
    best_seq, _ = max(beams, key=lambda x: x[1] / (max(1, x[0].size(1)) ** alpha))
    return best_seq

@torch.no_grad()
def topk_sampling(model, src, k=50, max_len=64, temperature=1.0):
    """
    Stochastic decoding: sample from top-k tokens at each step.
    """
    device = next(model.parameters()).device
    ys = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model(src, ys)                 # (B, T, V)
        logits = logits[:, -1, :] / max(1e-6, temperature)
        logp = torch.log_softmax(logits, dim=-1)  # (B, V)
        values, indices = torch.topk(logp, k, dim=-1)  # (B, k)
        # Sample from the top-k distribution
        choice = torch.distributions.Categorical(logits=values).sample()  # (B,)
        next_id = indices.gather(-1, choice.unsqueeze(-1))                # (B, 1)
        ys = torch.cat([ys, next_id], dim=1)
        if (next_id.squeeze(-1) == EOS_ID).all():
            break
    return ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (e.g., checkpoints/best.pt)")
    ap.add_argument("--src", required=True, help="Source sentence (space-tokenized)")
    ap.add_argument("--strategy", choices=["greedy", "beam", "topk"], default="greedy")
    ap.add_argument("--beam-size", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.7, help="Beam length norm exponent")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, src_vocab, tgt_vocab = load_ckpt(args.ckpt, args.device)
    src_ids = tokenize(src_vocab, args.src, max_len=args.max_len).to(args.device)

    if args.strategy == "greedy":
        out = greedy_decode(model, src_ids, max_len=args.max_len)
    elif args.strategy == "beam":
        out = beam_search(model, src_ids, beam_size=args.beam_size, max_len=args.max_len, alpha=args.alpha)
    else:
        out = topk_sampling(model, src_ids, k=args.topk, max_len=args.max_len, temperature=args.temperature)

    out_ids = out[0].tolist()
    print("Decoded token ids:", out_ids)
    print("Decoded tokens:", " ".join(tgt_vocab.decode(out_ids)))

if __name__ == "__main__":
    main()