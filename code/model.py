from encoder import Encoder
from decoder import Decoder
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_layers=4, num_heads=8, d_ff=1024, dropout=0.1, posenc="rope"):
        super().__init__()
        rope = posenc == "rope"
        rel_bias = posenc == "relbias"
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, rope=rope, rel_bias=rel_bias)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, rope=rope, rel_bias=rel_bias)
    def forward(self, src_ids, tgt_in_ids, src_mask=None, tgt_mask=None):
        mem = self.encoder(src_ids, src_mask=src_mask)
        logits = self.decoder(tgt_in_ids, mem, tgt_mask=tgt_mask)
        return logits