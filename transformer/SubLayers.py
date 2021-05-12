''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 一个可学习的矩阵[embedding_d, n_head * single_d]
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        # q和k计算attn时是做weight sum match，所以d_k=d_q
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # 残差是没有进行attention之前的值
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # 在普通attention机制中，q来自外部，v=k来自自身，attention = math(q, k)
        # 而在self-attention中，q=v=k都来自自身，那么，attention = math(q, k) 的结果最相似的，所以我们需要产生q != k，所以使用一个nn.Linear，一个可学习的矩阵[embedding_d, n_head * single_d]
        # 比如embedding后一个词表示为512维，q[seq_len， 512] * w[512, 8 * 64]，所以得到的query为8个(多头)64维向量，同理k和v
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # n_head 移到前面
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # 多头拆分了，从batch, seq_len, d_model, 到batch, seq_len, n_head, d_single_model(single_d)
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # 为什么多头要拆开？看attention里解释
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # 后面没有在涉及attn计算，合并多头便于计算
        q = self.dropout(self.fc(q))
        q += residual

        # channel方向做归一化
        q = self.layer_norm(q)

        # 输出attn，仅用于调试或可视化
        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
