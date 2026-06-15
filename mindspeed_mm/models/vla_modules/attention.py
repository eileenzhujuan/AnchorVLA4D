import torch
from torch import nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Q, K, and V
        self.w_q = nn.Linear(d_model, d_model) # From sequence A
        self.w_k = nn.Linear(d_model, d_model) # From sequence B
        self.w_v = nn.Linear(d_model, d_model) # From sequence B

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv):
        batch_size = x_q.size(0)

        # 1. Linear projections
        Q = self.w_q(x_q)
        K = self.w_k(x_kv)
        V = self.w_v(x_kv)

        # 2. Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        # Formula: Softmax( (Q @ K^T) / sqrt(d_k) ) @ V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attention_weights, V)

        # 4. Concatenate heads and pass through final linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.fc_out(out)
