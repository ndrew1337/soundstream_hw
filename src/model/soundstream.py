import torch.nn as nn
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.rvq import ResidualVectorQuantizer


class SoundStream(nn.Module):
    def __init__(
        self,
        channels=32,
        num_quantizers=8,
        num_embeddings=1024,
        embedding_dim=128,
        causal=True,
    ):
        super().__init__()
        self.causal = causal
        self.encoder = Encoder(channels, embedding_dim, causal=causal)
        self.decoder = Decoder(channels, embedding_dim, causal=causal)
        self.rvq = ResidualVectorQuantizer(num_quantizers, num_embeddings, embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.rvq(z)
        x_recon = self.decoder(z_q)
        if x_recon.shape[-1] != x.shape[-1]:
            x_recon = x_recon[..., : x.shape[-1]]
        return x_recon, vq_loss, indices
