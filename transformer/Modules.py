import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        # query和key进行match，这里match是直接相乘，其意义是query在每维key中的匹配情况
        # 除以sqrt{d}是为了防止维度过高内积值太大
        # 单头：q[seq_len, single_d] * k[single_d， seq_len]，其代表有seq_len个query[single_d]
        # 每个query和seq_len个key[single_d]计算match（也是做weight sum），所以1个query和seq_len个key计算得到attn[k_seq_len]
        # 而一共有seq_len个query，所以有[q_seq_len, k_seq_len]个attn

        #        q_s_d               k_l              k_l
        #    |—— —— —— ——|        |——|——|——|      |——|——|——|  // 单个query得到的attn[k_seq_len]
        # q_l|—— —— —— ——| x k_s_d|  |  |  | = q_l|——|——|——|
        #    |—— —— —— ——|        |  |  |  |      |——|——|——|
        #                         |——|——|——|
        # 解释为什么要把多头拆分：观察上面s_d是单头的维度，此时得到的基于单头的match(q_s_d, k_s_d)做weight sum得到一个数，所以每个头得到的atte是不一样的，意义就是学习不同类型的注意力
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # padding 部分设为极小，经过softmax后为0，表示忽略
            # 注意区分维度： encoder[1, src_seq_len]  decoder[trg_seq_len, trg_seq_len]
            # encoder时每个query（理解为从左往右对应的query）其对应的是完整的一个序列，所以[1, src_seq_len]的mask对应到每个query上，无差别
            # decoder时每个query（理解为从左往右对应的query）其对应的不是完整的一个序列，所以需要trg_seq_len个[1, trg_seq_len]的不同mask对应到每个query上
            # 补充一下decoder时每个query，使用mask的下三角形就能实现，如第一个query只有前1个词，第二个query有前2个词...，当然训练时可以这样批量放在矩阵操作，测试时则需要按照时间来逐个输出
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        # 应用attention
        # attn[q_seq_len, k_seq_len] * v[v_seq_len, single_d] 代表每个query计算得到的attn[k_seq_len]和v_seq_len个v[single_d]计算输出[single_d](每个维度都应用weight sum过程)
        # 所以seq_len个query得到的输出为[q_seq_len, single_d]

        #        k_l            v_s_d               v_s_d
        #    |——|——|——|      |—— —— —— ——|      |—— —— —— ——| // 再在每个维度上进行，得到v[single_d]
        # q_l|——|——|——| x v_l|—— —— —— ——| = q_l|—— —— —— ——|
        #    |——|——|——|      |—— —— —— ——|      |—— —— —— ——|
        #                                         \ 应用attn，然后weight sum过程

        # 注意q_s_d和k_s_d的维度一定要一样，用于匹配attn，应用attn时，是在v的每个维度上都应用attn，所以v_s_d不需要等于k_s_d或q_s_d
        output = torch.matmul(attn, v)

        # 输出attn，仅用于调试或可视化
        return output, attn
