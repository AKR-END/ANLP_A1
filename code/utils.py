import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

class SimpleVocab:
    def __init__(self, tokens: List[str]):
        uniq = ["<pad>", "<bos>", "<eos>"] + sorted(set(t for t in tokens if t not in ["<pad>", "<bos>", "<eos>"]))
        self.itos = uniq
        self.stoi = {t:i for i,t in enumerate(self.itos)}
    def __len__(self): return len(self.itos)
    def encode(self, toks: List[str]) -> List[int]:
        return [self.stoi.get(t, 0) for t in toks]
    def decode(self, ids: List[int]) -> List[str]:
        out = []
        for i in ids:
            if i == EOS_ID: break
            if i in (PAD_ID, BOS_ID): continue
            out.append(self.itos[i] if i < len(self.itos) else "<unk>")
        return out

class ParallelTextDataset(Dataset):
    def __init__(self, src_lines: List[str], tgt_lines: List[str], src_vocab: SimpleVocab, tgt_vocab: SimpleVocab, max_len: int = 128):
        assert len(src_lines) == len(tgt_lines)
        self.src = [l.strip().split()[:max_len-2] for l in src_lines]
        self.tgt = [l.strip().split()[:max_len-2] for l in tgt_lines]
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    def __len__(self): return len(self.src)
    def __getitem__(self, idx):
        s = ["<bos>"] + self.src[idx] + ["<eos>"]
        t = ["<bos>"] + self.tgt[idx] + ["<eos>"]
        s_ids = [self.src_vocab.stoi.get(tok, 0) for tok in s]
        t_ids = [self.tgt_vocab.stoi.get(tok, 0) for tok in t]
        return torch.tensor(s_ids, dtype=torch.long), torch.tensor(t_ids, dtype=torch.long)

def collate_pad(batch, pad_id=PAD_ID):
    src_seqs, tgt_seqs = zip(*batch)
    max_src = max(len(x) for x in src_seqs)
    max_tgt = max(len(x) for x in tgt_seqs)
    def pad(seqs, max_len):
        out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            out[i, :len(seq)] = seq
        return out
    return pad(src_seqs, max_src), pad(tgt_seqs, max_tgt), None, None

def corpus_bleu(refs: List[List[str]], hyps: List[List[str]], max_n:int=4, smooth:bool=True) -> float:
    from collections import Counter
    import math
    def _precisions(refs, hyps, n):
        numer, denom = 0, 0
        for ref, hyp in zip(refs, hyps):
            hyp_ngrams = [tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1)]
            ref_ngrams = [tuple(ref[i:i+n]) for i in range(len(ref)-n+1)]
            hyp_c, ref_c = Counter(hyp_ngrams), Counter(ref_ngrams)
            clipped = {g: min(cnt, ref_c.get(g,0)) for g,cnt in hyp_c.items()}
            numer += sum(clipped.values())
            denom += max(1, sum(hyp_c.values()))
        return numer / max(1, denom)
    precisions = []
    for n in range(1,max_n+1):
        p = _precisions(refs, hyps, n)
        if p==0 and smooth: p=1e-9
        precisions.append(p)
    geo = math.exp(sum(math.log(p) for p in precisions)/max_n)
    ref_len = sum(len(r) for r in refs)
    hyp_len = sum(len(h) for h in hyps)
    bp = 1.0 if hyp_len>ref_len else math.exp(1 - ref_len/max(1,hyp_len))
    return bp*geo

def set_seed(seed:int=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)