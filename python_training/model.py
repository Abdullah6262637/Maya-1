import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MayaConfig:
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # Grouped Query Attention (GQA)
    intermediate_size: int = 5632  # SwiGLU dimension
    rms_norm_eps: float = 1e-6
    max_seq_len: int = 2048
    initializer_range: float = 0.02
    use_mtp: bool = True
    lambda_mtp: float = 0.15

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex tensor
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"{freqs_cis.shape} != {(x.shape[1], x.shape[-1])}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convert to complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Apply rotation and flatten back to real numbers
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, config: MayaConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.hidden_size // self.n_heads

        self.wq = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.wo.weight._is_residual = True

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Expand Key/Value heads if they are grouped (GQA)
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # Reshape to (bs, heads, seqlen, dim) for SDPA
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Use PyTorch's native SDPA (Scaled Dot Product Attention)
        # It automatically dispatches to FlashAttention-2/3 or MemoryEfficientAttention when applicable
        is_causal = mask is None
        output = F.scaled_dot_product_attention(
            xq, xk, xv, 
            attn_mask=mask, 
            dropout_p=0.0, 
            is_causal=is_causal
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, config: MayaConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w2.weight._is_residual = True
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x) -> torch.Tensor:
        # SwiGLU activation
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: MayaConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN residual connections
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class MTPHead(nn.Module):
    """
    Multi-Token Prediction Head.
    Predicts token t+2 given hidden state h_t and embedding of token t+1.
    """
    def __init__(self, config: MayaConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.block = TransformerBlock(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, h: torch.Tensor, next_token_emb: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.cat([h, next_token_emb], dim=-1)
        x = self.proj(x)
        x = self.block(x, freqs_cis, mask)
        x = self.norm(x)
        return x

class MayaModel(nn.Module):
    def __init__(self, config: MayaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Shared embedding and output weight
        self.tok_embeddings.weight = self.output.weight

        # Precompute RoPE frequencies on CPU, will move to correct device during forward pass
        self.freqs_cis = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, 
            config.max_seq_len
        )
        
        # MTP Head Initialization
        if config.use_mtp:
            self.mtp_head = MTPHead(config)
            
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = self.config.initializer_range
            if getattr(module.weight, "_is_residual", False):
                std = std / math.sqrt(2 * self.config.num_hidden_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Retrieve RoPE frequencies for this seqlen
        freqs_cis = self.freqs_cis[:seqlen].to(tokens.device)

        # Forward pass through decoder layers
        for layer in self.layers:
            h = layer(h, freqs_cis)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        self.last_loss_dict = {}
        if targets is not None:
            # Main NTP Loss
            ntp_loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            loss = ntp_loss
            self.last_loss_dict["ntp_loss"] = ntp_loss.item()
            
            # Auxiliary MTP Loss
            if self.config.use_mtp and seqlen > 1:
                h_slice = h[:, :-1, :]
                targets_next = targets[:, :-1]
                targets_next_emb = self.tok_embeddings(targets_next)
                
                freqs_cis_slice = self.freqs_cis[:seqlen-1].to(tokens.device)
                
                # Forward pass through MTP head
                mtp_states = self.mtp_head(h_slice, targets_next_emb, freqs_cis_slice)
                logits_mtp = self.output(mtp_states)
                
                # Targets for MTP are tokens at step t+2 (which are y[:, 1:])
                targets_mtp = targets[:, 1:]
                
                mtp_loss = F.cross_entropy(logits_mtp.reshape(-1, self.vocab_size), targets_mtp.reshape(-1))
                
                loss = ntp_loss + self.config.lambda_mtp * mtp_loss
                self.last_loss_dict["mtp_loss"] = mtp_loss.item()
                self.last_loss_dict["total_loss"] = loss.item()

        return logits, loss

    def estimate_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
