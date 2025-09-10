import pandas as pd
import matplotlib.pyplot as plt

# Paths to your CSV files
rope_csv = "loss_log.csv"
relbias_csv = "rel_loss_log.csv"

# Read both CSVs
df_rope = pd.read_csv(rope_csv)
df_relbias = pd.read_csv(relbias_csv)

# Assume both CSVs have columns: epoch, train_loss, val_loss, test_loss
plt.figure(figsize=(8,6))

# Plot RoPE curves
plt.plot(df_rope["epoch"], df_rope["train_loss"], label="RoPE - Train", color="blue")
plt.plot(df_rope["epoch"], df_rope["val_loss"], label="RoPE - Val", color="orange")
plt.plot(df_rope["epoch"], df_rope["test_loss"], label="RoPE - Test", color="green")

# Plot RelBias curves
plt.plot(df_relbias["epoch"], df_relbias["train_loss"], "--", label="RelBias - Train", color="blue")
plt.plot(df_relbias["epoch"], df_relbias["val_loss"], "--", label="RelBias - Val", color="orange")
plt.plot(df_relbias["epoch"], df_relbias["test_loss"], "--", label="RelBias - Test", color="green")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves: RoPE vs RelBias")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curves_comparison.png", dpi=300)
print("Plot saved as loss_curves_comparison.png")