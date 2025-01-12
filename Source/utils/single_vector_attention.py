import torch
from torch import nn, Tensor
import math


class SingleVectorAttention(nn.Module):
    def __init__(self, embedding: int, heads: int):
        super().__init__()
        assert embedding % heads == 0, "Embedding dimension must be divisible by the number of attention heads."

        self.embedding = embedding
        self.heads = heads
        self.head_dim = embedding//heads
        self.scale = math.sqrt(self.head_dim)

        self.q = nn.Linear(embedding, embedding)
        self.k = nn.Linear(embedding, embedding)
        self.v = nn.Linear(embedding, embedding)
        self.out_proj = nn.Linear(embedding, embedding)
    
    def forward(self, x: Tensor) -> Tensor: # [batch, seq, log(seq), embedding]
        """
        Processes input using an attention mechanism to read from the logarithmic memory.

        This method is a placeholder for implementing an attention-based memory access
        scheme. It describes the intended interaction of an input `x` with the
        memory, using the logarithmic address in an attention mechanism.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len or 1, 1+ceil(log(seq_len))*bank, embedding_dim).

        Returns:
           Tensor: Output tensor of shape (batch_size, seq_len or 1, embedding_dim).
        """
        # Attention
        batch_size, seq, log_seq, embedding = x.shape

        mask = (x != 0) # mask zeros values to remain the same
        Q: Tensor = self.q(x) * mask
        K: Tensor = self.k(x[:, :, 0]) * (x[:, :, 0] != 0)
        V: Tensor = self.v(x) * mask

        Q = Q.view(batch_size, seq, log_seq, self.heads, self.head_dim).transpose(-2, -3)
        K = K.view(batch_size, seq, 1, self.heads, self.head_dim).transpose(-2, -3)
        V = V.view(batch_size, seq, log_seq, self.heads, self.head_dim).transpose(-2, -3) # [batch, seq, heads, log(seq), head_dim]

        scores = torch.einsum("...ij,...jk->...ik", Q, K.transpose(-1, -2)).squeeze(-1) # single vector attention #[batch, seq, heads, log(seq)]
        normalized_scores = scores / self.scale
        normalized_scores = torch.where(normalized_scores == 0, torch.full_like(normalized_scores, float('-inf')), normalized_scores) # mask attention
        attention_weights = torch.softmax(normalized_scores, dim=-1)
        
        attention_output = torch.einsum("...ki,...ij->...kj", attention_weights.unsqueeze(-2), V).squeeze(-2) # [batch, seq, heads, head_dim]
        attention_output = attention_output.reshape(batch_size, seq, embedding)
        output = self.out_proj(attention_output)

        return output
    
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)
    
