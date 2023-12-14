import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from .sam import StaticAttention
from einops import repeat
from einops import rearrange

from .embed import DSW_embedding


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
    def __init__(self, embed_dim, seq_length, feature_length, num_heads, dim_feedforward=None, d_values=[None, None], dropout=0.1,
                 batch_first=True):
        super(TSAEncoder, self).__init__()

        if dim_feedforward is None:
            dim_feedforward = [embed_dim * 4, seq_length]

        self.ttn = Encoder(d_model=embed_dim, nhead=num_heads[0], n_size=(seq_length, seq_length),
                           dim_feedforward=dim_feedforward[0], d_values=d_values[0], dropout=dropout,
                           batch_first=batch_first)
        self.sta = Encoder(d_model=embed_dim, nhead=num_heads[1], n_size=(feature_length, feature_length),
                           dim_feedforward=dim_feedforward[1], d_values=d_values[1], dropout=dropout,
                           batch_first=batch_first)

    def forward(self, data):
        output = data
        data = rearrange(data, 'b f l d -> (b f) l d')
        data = self.ttn(data)

        data = rearrange(data, '(b f) l d -> (b l) f d', b=output.shape[0])
        data = self.sta(data)

        data = rearrange(data, '(b l) f d -> b f l d', b=output.shape[0])
        output = output + data

        return output


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, seq_length=96, label_length=48, dropout=0.1, dim_feedforward=None, d_values=None,
                 activation=F.relu, batch_first=True):
        super(Decoder, self).__init__()

        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.decoder = nn.TransformerDecoderLayer(d_model=d_model,
                                                  nhead=nhead,
                                                  dim_feedforward=dim_feedforward,  #
                                                  dropout=dropout,
                                                  activation=activation,
                                                  batch_first=batch_first
                                                  )
        self.decoder.self_attn = StaticAttentionLayer(d_model, nhead, n_size=(label_length, label_length),
                                                      d_values=d_values, dropout=dropout)

    def forward(self, tgt, memory):
        batch_size = tgt.shape[0]
        tgt = rearrange(tgt, 'b f l d -> (b f) l d')
        memory = rearrange(memory, 'b f l d -> (b f) l d')

        tgt = self.decoder(tgt, memory)

        tgt = rearrange(tgt, '(b f) l d -> b f l d', b=batch_size)
        return tgt


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
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.dec_feature, (self.pred_len // self.seg_len), self.d_model))

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

        de_layers = OrderedDict()
        for i in range(self.num_de_layers):
            de_layers[f"decoder_layer_{i}"] = Decoder(
                d_model=self.d_model,
                nhead=self.nhead,
                label_length=(self.pred_len // self.seg_len),  ## self.label_len +
                seq_length=(self.seq_length // self.seg_len),
                dropout=dropout,
                batch_first=True
            )
        self.de_layers = nn.Sequential(de_layers)
        self.de_ln = nn.LayerNorm(self.d_model)

        heads_layers = OrderedDict()
        heads_layers["head"] = nn.Linear(self.d_model, self.seg_len, bias=True)

        self.heads = nn.Sequential(heads_layers)

        for layer in self.heads:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal(layer.weight)

    def forward(self, src, x_mark_enc, tgt, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        torch._assert(src.dim() == 3, f"src: Expected (batch_size, seq_length, hidden_dim) got {src.shape}")
        src = self.en_encoding(src)
        src += self.enc_pos_embedding
        src = self.pre_norm(src)

        for encoder in self.en_layers:
            src = encoder(src)
        src = self.en_ln(src)

        tgt = repeat(self.dec_pos_embedding, 'b f l d -> (repeat b) f l d', repeat=src.shape[0])

        for decoder in self.de_layers:
            tgt = decoder(tgt, src)
        tgt = self.de_ln(tgt)

        tgt = self.heads(tgt)
        res = rearrange(tgt, 'b f l seg_len -> b (l seg_len) f')
        if self.output_attention:
            return res[:, -self.pred_len:, :], None
        else:
            return res[:, -self.pred_len:, :]
