# train.py — with descriptive checkpoint filenames

import argparse, os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv
from typing import List, Tuple
from tqdm import tqdm

from utils import (
    ParallelTextDataset, SimpleVocab, collate_pad, set_seed,
    PAD_ID, BOS_ID, EOS_ID, corpus_bleu
)
from model import Transformer


def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def make_masks(src, tgt_in, pad_id=PAD_ID):
    src_mask = (src == pad_id).unsqueeze(1).unsqueeze(2).float() * -1e9
    tgt_mask = (tgt_in == pad_id).unsqueeze(1).unsqueeze(3).float() * -1e9
    return src_mask, tgt_mask


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    idx = torch.randperm(n, generator=gen).tolist()
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return train_idx, val_idx, test_idx


def build_vocab(lines: List[str]) -> SimpleVocab:
    toks = []
    for l in lines:
        toks.extend(l.strip().split())
    return SimpleVocab(toks)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad = 0
        self.stop = False

    def step(self, val_loss: float):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-src", type=str)
    ap.add_argument("--data-tgt", type=str)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--posenc", type=str, choices=["rope", "relbias"], default="rope")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", type=str, default="checkpoints")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--log-csv", type=str, default="loss_log.csv")
    ap.add_argument("--plot-png", type=str, default="loss_curve.png")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ===== Load dataset or toy =====
    if args.data_src and args.data_tgt:
        src_all = load_lines(args.data_src)
        tgt_all = load_lines(args.data_tgt)
    else:
        src_all = ["minä pidän sinusta","kissa on matolla","aurinko paistaa","miten voit"] * 50
        tgt_all = ["i like you","the cat is on the mat","the sun is shining","how are you"] * 50

    train_idx, val_idx, test_idx = split_indices(len(src_all), args.val_ratio, args.test_ratio, args.seed)
    src_train = [src_all[i] for i in train_idx]; tgt_train = [tgt_all[i] for i in train_idx]
    src_val   = [src_all[i] for i in val_idx];   tgt_val   = [tgt_all[i] for i in val_idx]
    src_test  = [src_all[i] for i in test_idx];  tgt_test  = [tgt_all[i] for i in test_idx]

    src_vocab, tgt_vocab = build_vocab(src_train), build_vocab(tgt_train)

    train_ds = ParallelTextDataset(src_train, tgt_train, src_vocab, tgt_vocab)
    val_ds   = ParallelTextDataset(src_val, tgt_val, src_vocab, tgt_vocab)
    test_ds  = ParallelTextDataset(src_test, tgt_test, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_pad)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_pad)

    model = Transformer(len(src_vocab), len(tgt_vocab),
                        d_model=args.d_model, num_layers=args.layers,
                        num_heads=args.heads, d_ff=args.d_ff,
                        dropout=args.dropout, posenc=args.posenc).to(args.device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    def train_epoch(loader, epoch, log_every=100):
        model.train()
        total = 0.0
        loop = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [train]", ncols=100)
        for i, (src, tgt, _, _) in loop:
            src, tgt = src.to(args.device), tgt.to(args.device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            src_mask, tgt_mask = make_masks(src, tgt_in)
            logits = model(src, tgt_in, src_mask.to(args.device), tgt_mask.to(args.device))
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()

            if (i + 1) % log_every == 0 or (i + 1) == len(loader):
                gpu_mem = torch.cuda.memory_allocated(args.device) / 1024**2 if torch.cuda.is_available() else 0
                msg = f"[train] epoch={epoch} batch={i+1}/{len(loader)} loss={loss.item():.4f} avg={total/(i+1):.4f} gpu_mem={gpu_mem:.0f}MB"
                print(msg, flush=True)

            loop.set_postfix(loss=loss.item(), avg=total/(i+1))

        return total / len(loader)


    def eval_loss(loader, epoch, split="val"):
        model.eval()
        total = 0.0
        loop = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [{split}]", ncols=100)
        with torch.no_grad():
            for i, (src, tgt, _, _) in loop:
                src, tgt = src.to(args.device), tgt.to(args.device)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                src_mask, tgt_mask = make_masks(src, tgt_in)
                logits = model(src, tgt_in, src_mask.to(args.device), tgt_mask.to(args.device))
                loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                total += loss.item()
                loop.set_postfix(loss=loss.item(), avg=total/(i+1))
        return total / len(loader)

    stopper=EarlyStopping(args.patience,args.min_delta)
    history={"epoch":[],"train":[],"val":[],"test":[]}
    best=float("inf")

    for e in range(1,args.epochs+1):
        tr = train_epoch(train_loader, e)
        va = eval_loss(val_loader, e, split="val")
        te = eval_loss(test_loader, e, split="test")
        print(f"==> Epoch {e} DONE: train={tr:.4f} val={va:.4f} test={te:.4f}", flush=True)

        history["epoch"].append(e); history["train"].append(tr)
        history["val"].append(va);  history["test"].append(te)

        ckpt={"epoch":e,"model_state":model.state_dict(),"opt_state":opt.state_dict(),
              "src_vocab":src_vocab.itos,"tgt_vocab":tgt_vocab.itos,"args":vars(args)}

        # save with descriptive name
        fname = f"{args.posenc}_epoch{e:03d}_train{tr:.4f}_val{va:.4f}.pt"
        torch.save(ckpt, os.path.join(args.save_dir, fname))

        if va < best - args.min_delta:
            best = va
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print("  ↳ New best model saved.")

        stopper.step(va)
        if stopper.stop:
            print("Early stopping triggered."); break

    # Save CSV
    csv_path=os.path.join(args.save_dir,args.log_csv)
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["epoch","train_loss","val_loss","test_loss"])
        for e,tr,va,te in zip(history["epoch"],history["train"],history["val"],history["test"]):
            w.writerow([e,tr,va,te])
    print(f"Saved CSV log to {csv_path}")

    # Plot
    plt.figure()
    plt.plot(history["epoch"], history["train"], label="train")
    plt.plot(history["epoch"], history["val"], label="val")
    plt.plot(history["epoch"], history["test"], label="test")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title(f"Loss curves ({args.posenc})")
    png_path=os.path.join(args.save_dir,args.plot_png)
    plt.savefig(png_path,bbox_inches="tight")
    print(f"Saved plot to {png_path}")


if __name__=="__main__":
    main()