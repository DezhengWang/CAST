import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from einops import rearrange

from models.Cast.embed import DSW_embedding
from models.Cast.static_attn import StaticAttention
from models.Cast.tools import Transpose


class StaticAttentionLayer(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, out_dim=None, n_size=(96, 96), d_values=None, mix=False, dropout=0.1):
        super(StaticAttentionLayer, self).__init__(embed_dim, num_heads)

        self.d_values = d_values or (embed_dim // num_heads)

        self.inner_attention = StaticAttention(dropout=dropout, n_size=n_size, nhead=num_heads)
        self.value_projection = nn.Linear(embed_dim, self.d_values * num_heads)
        if out_dim is None:
            out_dim = embed_dim
        self.out_projection = nn.Linear(self.d_values * num_heads, out_dim)
        self.n_heads = num_heads
        self.mix = mix

    def forward(self, query, key, value,
                key_padding_mask=None,
                need_weights=False,
                attn_mask=None,
                average_attn_weights=True,
                is_causal=False):

        B, L, _ = query.shape
        _, S, _ = value.shape
        H = self.n_heads

        values = self.value_projection(value).view(B, S, H, -1)

        out, attn = self.inner_attention(values, B, self.d_values)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        if need_weights:
            return self.out_projection(out), attn
        else:
            return self.out_projection(out), None


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, n_size=(96, 96), d_values=None, dropout=0.1, dim_feedforward=None,
                 activation=F.relu, batch_first=True):
        super(Encoder, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.encoder = nn.TransformerEncoderLayer(d_model=d_model,
                                                  nhead=nhead,
                                                  dim_feedforward=dim_feedforward,  #
                                                  dropout=dropout,
                                                  activation=activation,
                                                  batch_first=batch_first
                                                  )
        self.encoder.self_attn = StaticAttentionLayer(d_model, nhead, n_size=n_size, d_values=d_values, dropout=dropout)

    def forward(self, data):
        return self.encoder(data)


class TSAEncoder(nn.Module):
    def __init__(self, embed_dim, seq_length, feature_length, num_heads, dim_feedforward=None, d_values=[None, None],
                 dropout=0.1, batch_first=True):
        super(TSAEncoder, self).__init__()

        if dim_feedforward is None:
            dim_feedforward = [embed_dim * 4, seq_length]

        self.tta = Encoder(d_model=embed_dim, nhead=num_heads[0], n_size=(seq_length, seq_length),
                           dim_feedforward=dim_feedforward[0], d_values=d_values[0], dropout=dropout,
                           batch_first=batch_first)
        self.sta = Encoder(d_model=embed_dim, nhead=num_heads[1], n_size=(feature_length, feature_length),
                           dim_feedforward=dim_feedforward[1], d_values=d_values[1], dropout=dropout,
                           batch_first=batch_first)

        self.norm_ttn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(embed_dim), Transpose(1, 2))
        self.norm_sta = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(embed_dim), Transpose(1, 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        b = data.shape[0]
        tta_data = rearrange(data, 'b f l d -> (b f) l d')
        tta_data = self.tta(tta_data)
        tta_data = self.norm_ttn(tta_data)
        tta_data = rearrange(tta_data, '(b f) l d -> b f l d', b=b)

        sta_data = rearrange(data, 'b f l d -> (b l) f d')
        sta_data = self.sta(sta_data)
        sta_data = self.norm_sta(sta_data)
        sta_data = rearrange(sta_data, '(b l) f d -> b f l d', b=b)

        output = data + sta_data + tta_data

        return output

"""
Transformer + Cross Attention + DSG + Static Attention + Prediction Head + Non-stationary
"""
class CAST(nn.Module):
    def __init__(self, args, **factory_kwargs):
        super(CAST, self).__init__()
        self.enc_feature = args.enc_in
        self.dec_feature = args.dec_in
        self.out_feature = args.c_out
        self.seq_length = args.size[0]
        self.label_len = args.size[1]
        self.pred_len = args.iter_horizon

        self.d_model = args.d_model
        self.nhead = args.nhead
        self.num_en_layers = args.e_layers
        self.num_de_layers = args.d_layers
        self.seg_len = getattr(args, 'seg_len', 6)  # default 6

        dropout = args.dropout
        self.output_attention = args.output_attention

        self.en_encoding = DSW_embedding(self.seg_len, self.d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.enc_feature, (self.seq_length // self.seg_len), self.d_model))
        self.pre_norm = nn.LayerNorm(self.d_model)

        self.dropout = nn.Dropout(dropout)

        en_layers = OrderedDict()
        for i in range(self.num_en_layers):
            en_layers[f"encoder_layer_{i}"] = TSAEncoder(
                embed_dim=self.d_model,
                feature_length=self.enc_feature,
                num_heads=[self.nhead, self.nhead],
                dim_feedforward=[self.d_model*4, self.d_model*4],
                seq_length=(self.seq_length // self.seg_len),
                dropout=dropout,
                batch_first=True
            )
        self.en_layers = nn.Sequential(en_layers)
        self.en_ln = nn.LayerNorm(self.d_model)

        heads_layers = OrderedDict()

        heads_layers["flatten"] = nn.Flatten(start_dim=-2)
        heads_layers["heads"] = nn.Linear((self.seq_length // self.seg_len)*self.d_model, self.pred_len)
        heads_layers["dropout"] = nn.Dropout(0.)

        self.heads = nn.Sequential(heads_layers)

        for layer in self.heads:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal(layer.weight)

    def forward(self, src, x_mark_enc, tgt, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        torch._assert(src.dim() == 3, f"src: Expected (batch_size, seq_length, hidden_dim) got {src.shape}")

        means = src.mean(1, keepdim=True).detach()
        src = src - means
        stdev = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5)
        src /= stdev

        src = self.en_encoding(src)
        src += self.enc_pos_embedding
        src = self.pre_norm(src)

        for encoder in self.en_layers:
            src = encoder(src)
        src = self.en_ln(src)

        res = self.heads(src)
        res = rearrange(res, 'b f l -> b l f')

        res = res * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        res = res + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        if self.output_attention:
            return res, None
        else:
            return res
