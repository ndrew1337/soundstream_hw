import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, iterations=50):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.requires_grad_(False)
        nn.init.uniform_(self.embeddings.weight, a=0.0, b=1.0)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embedding_avg", self.embeddings.weight.clone())
        self.register_buffer("initialized", torch.tensor(False))
        self.commitment_weight = 1.0
        self.decay = 0.99
        self.dead_code_threshold = 2.0
        self.eps = 1e-3
        self.iterations = iterations

    def forward(self, x_enc):
        x_flat = x_enc.transpose(1, 2).reshape(-1, self.embedding_dim)
        x_flat_fp = x_flat.float()
        weight_dtype = self.embeddings.weight.dtype

        if not self.initialized.item() and self.training:
            with torch.no_grad():
                num_vectors = x_flat_fp.shape[0]
                if num_vectors >= self.num_embeddings:
                    indices = torch.randperm(num_vectors, device=x_flat_fp.device)[:self.num_embeddings]
                else:
                    indices = torch.randint(0, num_vectors, (self.num_embeddings,), device=x_flat_fp.device)
                centers = x_flat_fp[indices]

                for i in range(self.iterations):
                    distances = torch.cdist(x_flat_fp, centers, p=2)
                    assignments = torch.argmin(distances, dim=1)
                    encodings = F.one_hot(assignments, num_classes=self.num_embeddings).float()
                    counts = encodings.sum(dim=0)
                    sums = encodings.T @ x_flat_fp
                    new_centers = sums / counts.clamp(min=1).unsqueeze(-1)
                    mask = (counts > 0).unsqueeze(-1)
                    centers = torch.where(mask, new_centers, centers)

                self.embeddings.weight.data.copy_(centers.to(weight_dtype))
                self.embedding_avg.data.copy_(centers.to(self.embedding_avg.dtype))
                self.cluster_size.data.fill_(1.0)
                self.initialized.fill_(True)

        distances = torch.cdist(x_flat, self.embeddings.weight.to(x_flat.dtype), p=2)
        encoding_indices = torch.argmin(distances, dim=1)
        x_q = self.embeddings(encoding_indices).to(x_flat.dtype)
        loss = F.mse_loss(x_flat, x_q.detach()) * self.commitment_weight
        x_q = x_flat + (x_q - x_flat).detach()

        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
                counts = encodings.sum(0)
                sums = encodings.T @ x_flat_fp

                self.cluster_size.data.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.embedding_avg.data.mul_(self.decay).add_(sums.to(self.embedding_avg.dtype), alpha=1 - self.decay)

                n = self.cluster_size.sum()
                cluster_size_smoothed = (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
                new_weight = self.embedding_avg / cluster_size_smoothed.unsqueeze(-1)
                self.embeddings.weight.data.copy_(new_weight.to(weight_dtype))

                dead = self.cluster_size < self.dead_code_threshold
                if dead.any():
                    n_dead = int(dead.sum().item())
                    n_avail = x_flat_fp.shape[0]
                    if n_avail >= n_dead:
                        rand_idx = torch.randperm(n_avail, device=x_flat_fp.device)[:n_dead]
                    else:
                        rand_idx = torch.randint(0, n_avail, (n_dead,), device=x_flat_fp.device)
                    self.embeddings.weight.data[dead] = x_flat_fp[rand_idx].to(weight_dtype)
                    self.embedding_avg.data[dead] = x_flat_fp[rand_idx].to(self.embedding_avg.dtype)
                    self.cluster_size.data[dead] = 1.0

        B, D, T = x_enc.shape
        x_q = x_q.reshape(B, T, self.embedding_dim).transpose(1, 2)
        return x_q, loss, encoding_indices.reshape(B, T)
                    

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_quantizers, num_embeddings, embedding_dim):
        super().__init__()
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim)
            for _ in range(num_quantizers)
        ])

    def forward(self, x_enc):
        residual = x_enc
        q_total = torch.zeros_like(x_enc)
        loss_total = 0.0
        all_indices = []
        for vq in self.quantizers:
            q_i, loss_i, idx_i = vq(residual)
            q_total = q_total + q_i
            residual = residual - q_i
            loss_total = loss_total + loss_i
            all_indices.append(idx_i)
        return q_total, loss_total, torch.stack(all_indices, dim=1)