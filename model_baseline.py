import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.nn import functional as F
from einops import rearrange

laynorm_eps = 1e-12

class GPTConfig_ANN(object):
    def __init__(self, 
                 hidden_dim=512,
                 vocab_size=3000,
                 block_size=128,
                 num_heads=8,
                 depths=2,
                 ):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_heads = num_heads
        self.depths = depths

class WordEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.ln = nn.LayerNorm(config.hidden_dim, eps=laynorm_eps)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.emb(x)  # B L D
        x = self.ln(self.fc(x))
        x = self.act(x)
        return x  # B L D

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.num_heads == 0, f"dim {config.hidden_dim} should be divided by num_heads {config.num_heads}."
        self.dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.scale = 0.125

        self.qkv_fc = nn.Linear(config.hidden_dim, config.hidden_dim * 3)
        self.ln = nn.LayerNorm(config.hidden_dim, eps=laynorm_eps)
        
        self.attn_act = nn.GELU()
        self.fc = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim, eps=laynorm_eps)
        
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv_fc(x).reshape(B, L, 3, D).permute(2, 0, 1, 3)  # qkv = [3, B, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, L, D]

        q = self.ln(q)
        k = self.ln(k)
        v = self.ln(v)

        q = q.reshape(B, L, self.num_heads, D//self.num_heads).transpose(1, 2)
        k = k.reshape(B, L, self.num_heads, D//self.num_heads).transpose(1, 2)
        v = v.reshape(B, L, self.num_heads, D//self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, L, D)
        x = self.attn_act(x)
        x = self.ln2(self.fc(x))
        
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim * 4)
        self.ln1 = nn.LayerNorm(config.hidden_dim * 4, eps=laynorm_eps)
        self.act1 = nn.GELU()
        
        self.fc2 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim, eps=laynorm_eps)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.act1(self.ln1(self.fc1(x)))
        x = self.act2(self.ln2(self.fc2(x)))
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class GPT_ANN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.emb = WordEmbedding(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depths)])
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def get_block_size(self):
        return self.block_size

    def forward(self, x, targets=None):
        x = self.emb(x)
        
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)

        logits = self.head(states[-1])  # B L vocab_size

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss 