import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    def __init__(self, dim=256):
        super(CrossAttention, self).__init__()
        self.dim = dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, img_feats, txt_feats):

        B, N_img, C = img_feats.shape
        _, N_txt, _ = txt_feats.shape


        Q = self.q_proj(img_feats)
        K = self.k_proj(txt_feats)
        V = self.v_proj(txt_feats)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        ctx = torch.matmul(attn, V)

        fused = self.out_proj(ctx)
        fused = fused + img_feats

        return fused