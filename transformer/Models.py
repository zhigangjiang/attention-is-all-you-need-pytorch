''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        # 设计这个位置函数的用意是位置可以由上一个位置线性表示，参考：https://zhuanlan.zhihu.com/p/308301901#1.4.1
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # x.shape = [batch_size, seq_len, embedding_d])
        # pos_table计算最大长度为200的位置编码，这里去前seq_len长度就行，在每个seq上加上[seq_len, embedding_d]的位置编码，相当于每个位置都加[embedding_d]
        # 注意: 这里这是数值上相加，不改变shape
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        # 训练一个Embedding层，把数据集中（输入数据）n_src_vocab维，压缩为d_word_vec(d_model)维
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        # 位置编码器，一个[d_word_vec, n_position]表
        # n_position表示一个足够长的序列，首先序列会补padding，和batch中最大长度sql_len一样长，n_position-sql_len部分就使用mask置0
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []
        # -- Forward
        # src_seq是从word到int，最大到n_src_vocab，所以对0~n_src_vocab，进行编码，每个词编码为d_word_vec(d_model)维
        # 比如：开始符号的整数编码为2，那么embedding后的编码为[0, 0, 1, 0, ... ,0]，看做n_src_vocab+1分类
        enc_output = self.src_word_emb(src_seq)
        # 注意此时enc_output的维度为d_word_vec，后面输入enc_layer，而enc_layer要求维度为d_model，所以必须d_word_vec == d_model
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5

        # 加上位置编码
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)
        for enc_layer in self.layer_stack:  # 有n_layers=6层，每层有n_head=8个头
            # 进入Encoder层，输出enc_output作为下层的输入
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            # 调试或者可视化使用
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()
        # 训练一个Embedding层，把数据集中（标注数据）n_trg_vocab维，压缩为d_word_vec(d_model)维
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        # 位置编码器，一个[d_word_vec, n_position]表
        # n_position表示一个足够长的序列，首先序列会补padding，和batch中最大长度sql_len一样长，n_position-sql_len部分就使用mask置0
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        # 训练时，标注数据和encoder输出 作为decoder的输入，而在测试时，上一时刻的decoder的最终输出和encoder输出 作为下一时刻的decoder输入
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            # 进入Decoder层，enc_output用于cross—attention，输出dec_output作为下层的输入
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            # 调试或者可视化使用
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab,  # 数据集中词的个数，比如10000个，那么就是[0~9999]中不同词，转成one-hot分类编码一个词就有10000维，显然可以使用embedding进行压缩
            n_position=n_position,  # 由于每个batch序列长度不一样，所以给个最大长度，我认为可以求数据集中序列最大长度作为n_position
            d_word_vec=d_word_vec,  # 经过embedding压缩后的维度，且作为后续EncoderLayer层的输入，所以要==d_model
            d_model=d_model,  # d_word_vec == d_model
            d_inner=d_inner,  # 前反馈层中间fc层维度，比如: d_model x d_inner x d_model，经过这3层，对一个EncoderLayer做最后处理
            n_layers=n_layers,  # 有多少EncoderLayer
            n_head=n_head,  # 多头注意力，每头注意力侧重点不一样，可以看到后面对多头进行拆分单头注意力分别计算，
            d_k=d_k,  # 注意query和key的维度一定要一样，用于匹配attn，所以在attention时d_k=d_q
            d_v=d_v,  # value维度，默认和k和v一样
            pad_idx=src_pad_idx,  # Embedding Layer 用，猜测padding需要特殊处理，所以需要先知道其编码(索引)
            dropout=dropout,
            scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab,  # 目标词个数
            n_position=n_position,  # 和encoder一样
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            pad_idx=trg_pad_idx,
            dropout=dropout,
            scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)  # mask作用为将padding的attn置0
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)  # 当解码第一个字的时候，第一个字只能与第一个字计算相关性，当解出第二个字的时候，只能计算出第二个字与第一个字和第二个字的相关性
        # 注意shape：src_mask[batch_size, 1, src_seq_len] trg_mask[batch_size, trg_seq_len, trg_seq_len]
        enc_output, *_ = self.encoder(src_seq, src_mask)  # enc_output用于decoder中的cross-attention
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)

        # dec_output的输出表示了seq上每个词的embedding向量（维度d_model），通过一个fc，转为词分类概率（维度n_trg_vocab）
        seq_logit = self.trg_word_prj(dec_output)
        # softmax放在外边的loss函数中了 :)

        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))  # 合并batch
