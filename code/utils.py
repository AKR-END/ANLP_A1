import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
try:
    import sentencepiece as spm
except ImportError:
    spm = None

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

class SentencePieceVocab:
    def __init__(self, model_proto: Optional[bytes] = None, model_file: Optional[str] = None):
        assert spm is not None, "sentencepiece is not installed. Add it to dependencies."
        self.sp = spm.SentencePieceProcessor()
        if model_proto is not None:
            # Load from serialized model bytes
            self.sp.LoadFromSerializedProto(model_proto)
        elif model_file is not None:
            self.sp.Load(model_file)
        else:
            raise ValueError("Provide either model_proto bytes or model_file path")
        # Keep a flag for dataset to branch
        self.is_spm = True
        self.offset = 3  # reserve 0:PAD, 1:BOS, 2:EOS

    def __len__(self):
        return int(self.sp.GetPieceSize()) + self.offset

    def encode_sentence(self, text: str, add_bos_eos: bool = True, max_len: Optional[int] = None) -> List[int]:
        ids = [i + self.offset for i in self.sp.EncodeAsIds(text)]
        if add_bos_eos:
            ids = [BOS_ID] + ids + [EOS_ID]
        if max_len is not None:
            ids = ids[:max(0, max_len)]
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        # Convert to words for BLEU comparable with word-split refs
        # Remove special tokens and stop at EOS
        clean = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            if i >= self.offset:
                clean.append(i - self.offset)
        text = self.sp.DecodeIds(clean)
        return text.strip().split()

class ParallelTextDataset(Dataset):
    def __init__(self, src_lines: List[str], tgt_lines: List[str], src_vocab, tgt_vocab, max_len: int = 128):
        assert len(src_lines) == len(tgt_lines)
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    def __len__(self): return len(self.src_lines)
    def __getitem__(self, idx):
        if hasattr(self.src_vocab, "is_spm") and getattr(self.src_vocab, "is_spm"):
            s_ids = self.src_vocab.encode_sentence(self.src_lines[idx].strip(), add_bos_eos=True, max_len=self.max_len)
        else:
            src_toks = self.src_lines[idx].strip().split()[: self.max_len - 2]
            s = ["<bos>"] + src_toks + ["<eos>"]
            s_ids = [self.src_vocab.stoi.get(tok, 0) for tok in s]

        if hasattr(self.tgt_vocab, "is_spm") and getattr(self.tgt_vocab, "is_spm"):
            t_ids = self.tgt_vocab.encode_sentence(self.tgt_lines[idx].strip(), add_bos_eos=True, max_len=self.max_len)
        else:
            tgt_toks = self.tgt_lines[idx].strip().split()[: self.max_len - 2]
            t = ["<bos>"] + tgt_toks + ["<eos>"]
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