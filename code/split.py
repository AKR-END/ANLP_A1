import torch

def split_dataset(src_all, tgt_all, val_ratio=0.1, test_ratio=0.1, seed=42, mode="perm"):
    """
    Split parallel dataset into train/val/test.
    
    Args:
        src_all, tgt_all: lists of strings (parallel data)
        val_ratio, test_ratio: floats (fractions)
        seed: int, RNG seed for permutation
        mode: "perm" = random permutation; "head" = first slices
    
    Returns:
        (src_train, tgt_train), (src_val, tgt_val), (src_test, tgt_test)
    """
    assert len(src_all) == len(tgt_all), "Source and target sizes differ"
    n = len(src_all)
    n_val, n_test = int(n * val_ratio), int(n * test_ratio)

    if mode == "perm":
        gen = torch.Generator(); gen.manual_seed(seed)
        idx = torch.randperm(n, generator=gen).tolist()
    else:  # head
        idx = list(range(n))

    val_idx  = idx[:n_val]
    test_idx = idx[n_val:n_val+n_test]
    train_idx= idx[n_val+n_test:]

    src_val  = [src_all[i] for i in val_idx]
    tgt_val  = [tgt_all[i] for i in val_idx]
    src_test = [src_all[i] for i in test_idx]
    tgt_test = [tgt_all[i] for i in test_idx]
    src_train= [src_all[i] for i in train_idx]
    tgt_train= [tgt_all[i] for i in train_idx]

    print(f"[split] total={n}  train={len(src_train)}  val={len(src_val)}  test={len(src_test)}", flush=True)

    return (src_train, tgt_train), (src_val, tgt_val), (src_test, tgt_test)

if __name__ == "__main__":
    with open("../data/EUbookshop.en", encoding="utf-8") as f: src_all = f.read().splitlines()
    with open("../data/EUbookshop.fi", encoding="utf-8") as f: tgt_all = f.read().splitlines()
    (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt) = split_dataset(src_all, tgt_all, val_ratio=0.05, test_ratio=0.05, seed=42, mode="perm")