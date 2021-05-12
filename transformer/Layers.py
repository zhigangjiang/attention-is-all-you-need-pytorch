''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # 由于q和k的维度默一定要一样，这里区分k和v的维度
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        # Encoder只有self-attention，即q=v=k，在Encoder中只有slf_attn，输入enc_input_i来自前一个输出enc_output_i-1
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)

        # 上面说到当前输出enc_output_i会作为下一层的输入enc_input_i+1，但是不能直接传过去，加一个前馈神经网络层pos_ffn
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 由于q和k的维度默一定要一样，这里区分k和v的维度
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        # cross-attention 用于提取src上对应当前时刻输出的注意力
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        # self-attention，即q=v=k，输入dec_input_i来自前一个输出dec_output_i-1
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        # 输出dec_output_i，作为cross-attention的query

        # cross-attention，即q!=v=k，其中query来自self-attention的输出（由于输出是一个一个输出的，所以在decoder上的self-attention不能够从src中提取注意力），
        # 而注意力是基于全局的，此时应用和query匹配的是具有全局完整信息的encoder上的key，然后应用到encoder上的value，用于提取src上对应当前时刻输出的注意力
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)

        # 和encoder同理不能直接传过去，加一个前馈神经网络层pos_ffn
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
