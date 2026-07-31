import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from mamba_ssm import Mamba


# ==================== RoPE  ====================

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).cuda()
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    batch_size, seq_len, dim = xq.shape
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    xq_out = torch.view_as_real(xq_ * freqs_cis[:seq_len]).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:seq_len]).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ==================== RoPE Attention ====================

class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.wq = nn.Linear(in_dim, out_dim)
        self.wk = nn.Linear(in_dim, out_dim)
        self.wv = nn.Linear(in_dim, out_dim)
        self.freqs_cis = precompute_freqs_cis(out_dim, 1500 * 2)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, seq_len, -1)
        xk = xk.view(batch_size, seq_len, -1)
        xv = xv.view(batch_size, seq_len, -1)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis)
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(dim)
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, xv)
        return output

# ==================== Traditional ====================

class TraditionalFeatureEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.aa_embed = nn.Embedding(22, out_dim, padding_idx=21)
        self.ss_aa_embed = nn.Embedding(64, out_dim, padding_idx=0)
        self.donor_proj = nn.Sequential(nn.Linear(1, out_dim), nn.LayerNorm(out_dim), nn.ReLU())
        self.acceptor_proj = nn.Sequential(nn.Linear(1, out_dim), nn.LayerNorm(out_dim), nn.ReLU())
        self.disorder_proj = nn.Sequential(nn.Linear(3, out_dim), nn.LayerNorm(out_dim), nn.ReLU())
        self.pp_proj = nn.Sequential(nn.Linear(7, out_dim), nn.LayerNorm(out_dim), nn.ReLU())
        self.rsa_proj = nn.Sequential(nn.Linear(1, out_dim), nn.LayerNorm(out_dim), nn.ReLU())

        concat_dim = out_dim * 7
        self.cnn = nn.Sequential(
            nn.Conv1d(concat_dim, concat_dim, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(concat_dim, concat_dim, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(concat_dim, concat_dim, kernel_size=5, padding=2), nn.ReLU()
        )
        self.output_norm = nn.LayerNorm(concat_dim)
        self.out_dim = concat_dim

    def forward(self, aa_indices, ss_aa_codes, donor, acceptor, disorder, pp, rsa):
        aa_enc = self.aa_embed(aa_indices.long())
        ss_enc = self.ss_aa_embed(ss_aa_codes.long())
        donor_enc = self.donor_proj(donor.float())
        acceptor_enc = self.acceptor_proj(acceptor.float())
        disorder_enc = self.disorder_proj(disorder.float())
        pp_enc = self.pp_proj(pp.float())
        rsa_enc = self.rsa_proj(rsa.float())

        fused = torch.cat([aa_enc, ss_enc, donor_enc, acceptor_enc, disorder_enc, pp_enc, rsa_enc], dim=-1)
        fused = fused.permute(0, 2, 1)
        cnn_out = self.cnn(fused)
        fused = fused + cnn_out
        fused = fused.permute(0, 2, 1)
        fused = self.output_norm(fused)
        return fused

# ==================== FusionTransformer ====================

class FusionTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.freqs_cis = precompute_freqs_cis(self.head_dim, 1500 * 2)
        ffn_dim = dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn_drop = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        B, H, L, hd = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * hd)

    def _apply_rope_multihead(self, xq, xk):
        B, H, L, hd = xq.shape
        xq_ = xq.reshape(B * H, L, hd)
        xk_ = xk.reshape(B * H, L, hd)
        xq_out, xk_out = apply_rotary_emb(xq_, xk_, freqs_cis=self.freqs_cis)
        return xq_out.view(B, H, L, hd), xk_out.view(B, H, L, hd)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        xq = self._split_heads(self.q_proj(x))
        xk = self._split_heads(self.k_proj(x))
        xv = self._split_heads(self.v_proj(x))
        xq, xk = self._apply_rope_multihead(xq, xk)
        attn = torch.matmul(xq, xk.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn.float(), dim=-1).type_as(xq)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, xv)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x

class FusionTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, num_heads=8, ffn_ratio=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        self.layers = nn.ModuleList(
            [FusionTransformerBlock(out_dim, num_heads, ffn_ratio, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# ==================== Mamba ====================

class MambaBlock(nn.Module):
    def __init__(self, dim: int, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return x + self.mamba(self.norm(x))

class MambaEncoder(nn.Module):
    def __init__(self, dim, num_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(dim, d_state, d_conv, expand) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

# ==================== JointPeptide ====================

class JointPeptide(nn.Module):
    def __init__(self, in_dim, out_dim, ws, mamba_layers=2):
        super().__init__()
        pretrained_dict = torch.load("./pkls/esm2_t33_cls_new.pkl")["dipeptides"].view(484, ws - 1, -1)
        self.dipeptides = nn.Parameter(pretrained_dict, requires_grad=True)
        self.ws = ws
        self.indices_y = torch.tensor([_ for _ in range(self.ws - 1)], device="cuda")
        self.attention = Attention(in_dim, out_dim)
        self.mamba_encoder = MambaEncoder(
            dim=out_dim, num_layers=mamba_layers, d_state=16, d_conv=4, expand=2, dropout=0.5
        )

    def get_window_index(self, ids):
        indice = []
        for x in range(self.ws - 1):
            indice.append(ids[x] * 22 + ids[x + 1])
        return torch.tensor(indice, device="cuda")

    def sliding_window(self, x):
        ids = x[0, 1:-1] - 4
        ids = torch.cat([torch.full([self.ws // 2], 21, device="cuda"), ids])
        ids = torch.cat([ids, torch.full([self.ws // 2], 21, device="cuda")])
        indices = []
        for i in range(0, len(ids) - self.ws + 1):
            win_ids = ids[i:i + self.ws]
            indice = self.get_window_index(win_ids)
            indices.append(indice)
        indices = torch.stack(indices)
        return indices.long()

    def forward(self, x):
        indices = self.sliding_window(x)
        x = self.dipeptides[indices, self.indices_y].view(-1, 1280).unsqueeze(0)
        x = self.attention(x)
        x = self.mamba_encoder(x)
        return x





class PepMultiFuse(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model_name_or_path)
        if "esm" in args.model_name_or_path:
            with torch.no_grad():
                old_pos = self.model.embeddings.position_embeddings.weight.clone().detach()
                new_pos = self.model.embeddings.position_embeddings_test.weight.clone().detach()
                alpha = torch.tensor(0.4)
                for j in range(1500):
                    x, y = j // 1024 + 1, j % 1024
                    new_pos[j] = alpha * old_pos[x] + (1 - alpha) * old_pos[y]
            self.model.embeddings.position_embeddings_test.weight = nn.Parameter(
                new_pos.clone().detach().requires_grad_(True)
            )
            args.model_dim = 640 if "t30" in args.model_name_or_path else 1280

        self.dim = 128
        self.traditional_encoder = TraditionalFeatureEncoder(out_dim=64)
        self.joint_peptide_both = JointPeptide(in_dim=1280, out_dim=128, ws=args.ws, mamba_layers=2)


        self.spatial_proj = nn.Sequential(
            nn.Linear(3840, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
        )
        self.spatial_gate = nn.Parameter(torch.tensor(-2.0))


        fusion_in_dim = args.model_dim + 128 + 448 + 128

        self.fusion_transformer = FusionTransformer(
            in_dim=fusion_in_dim, out_dim=self.dim,
            num_layers=2, num_heads=8, ffn_ratio=4, dropout=0.1
        )
        self.fnn = nn.Sequential(
            nn.Linear(self.dim, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, args.logit, bias=True)
        )

    def forward(self, x,
                aa_indices=None, ss_aa_codes=None,
                donor=None, acceptor=None,
                disorder=None, pp=None, rsa=None,
                precomputed_spatial=None, spatial_mask=None,
                logits=True):

        # 1) ESM2
        model_out = self.model(x).last_hidden_state[:, 1:-1]

        # 2) JointPeptide
        out_peptide = self.joint_peptide_both(x)

        # 3) Traditional
        if aa_indices is not None and ss_aa_codes is not None:
            traditional_out = self.traditional_encoder(
                aa_indices, ss_aa_codes, donor, acceptor, disorder, pp, rsa)
        else:
            traditional_out = torch.zeros(
                model_out.shape[0], model_out.shape[1], 448,
                device=model_out.device, dtype=model_out.dtype)

        # 4) spatial
        if precomputed_spatial is not None and spatial_mask is not None:
            spatial = precomputed_spatial.to(device=model_out.device, dtype=model_out.dtype)
            mask = spatial_mask.to(device=model_out.device, dtype=model_out.dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            spatial_feat = self.spatial_proj(spatial)
            gate = torch.sigmoid(self.spatial_gate)
            spatial_out = spatial_feat * mask * gate
        else:
            spatial_out = torch.zeros(
                model_out.shape[0], model_out.shape[1], 128,
                device=model_out.device, dtype=model_out.dtype)


        min_len = min(model_out.shape[1], out_peptide.shape[1],
                      traditional_out.shape[1], spatial_out.shape[1])
        model_out = model_out[:, :min_len]
        out_peptide = out_peptide[:, :min_len]
        traditional_out = traditional_out[:, :min_len]
        spatial_out = spatial_out[:, :min_len]


        fused = torch.cat([model_out, out_peptide, traditional_out, spatial_out], dim=-1)
        out = self.fusion_transformer(fused)

        if logits:
            return self.fnn(out)[0]
        else:
            out_feature = self.fnn[:3](out)
            preds = self.fnn[3:](out_feature)
            return out[0], out_feature[0], preds[0]