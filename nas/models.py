"""
Neural Architecture Implementations

各アーキテクチャの実際の実装:
- Transformer (standard attention)
- Linear Transformer (linear attention)
- FlashAttention (memory efficient)
- Grouped Query Attention (GQA)
- Mamba (state space model)
- RWKV (RNN-like transformer)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from search_space import ArchitectureConfig


# ===== Normalization Layers =====

def get_normalization(norm_type: str, dim: int) -> nn.Module:
    """Get normalization layer"""
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim)
    elif norm_type == "groupnorm":
        # Default: 8 groups
        num_groups = min(8, dim)
        return nn.GroupNorm(num_groups, dim)
    else:
        raise ValueError(f"Unknown normalization: {norm_type}")


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Used in: LLaMA, Gopher, Chinchilla
    Paper: "Root Mean Square Layer Normalization" (2019)

    Faster than LayerNorm (no mean subtraction)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm


# ===== Activation Functions =====

def get_activation(act_type: str) -> nn.Module:
    """Get activation function"""
    if act_type == "gelu":
        return nn.GELU()
    elif act_type == "silu":
        return nn.SiLU()
    elif act_type == "geglu":
        return GeGLU()
    elif act_type == "swiglu":
        return SwiGLU()
    else:
        raise ValueError(f"Unknown activation: {act_type}")


class GeGLU(nn.Module):
    """
    GELU Gated Linear Unit

    Paper: "GLU Variants Improve Transformer" (2020)
    Used in: GPT-3, PaLM

    Formula: GELU(x @ W) * (x @ V)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class SwiGLU(nn.Module):
    """
    Swish/SiLU Gated Linear Unit

    Paper: "GLU Variants Improve Transformer" (2020)
    Used in: LLaMA, PaLM 2

    Formula: SiLU(x @ W) * (x @ V)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


# ===== Position Encoding =====

class PositionalEncoding(nn.Module):
    """
    Absolute sinusoidal positional encoding

    Paper: "Attention is All You Need" (2017)
    Used in: Original Transformer, BERT, GPT
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
    Used in: LLaMA, GPT-NeoX, PaLM

    Advantages:
    - Relative position encoding
    - Better long-range modeling
    - No trainable parameters
    """

    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()

        # Precompute rotation matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cos, sin for rotary embedding
        """
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len

            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

        return self._cos_cached, self._sin_cached


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding"""
    # Rotate q and k
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# ===== Attention Mechanisms =====

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention

    Paper: "Attention is All You Need" (2017)
    Complexity: O(n^2 * d)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = False
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len, seq_len) or None

        Returns:
            (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv(x)  # (B, L, 3*H)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, L, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, nh, L, hd)

        # Apply RoPE if enabled
        if self.use_rope:
            cos, sin = self.rotary_emb(x, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, L, L)

        # Apply mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Combine heads
        out = attn @ v  # (B, nh, L, hd)
        out = out.transpose(1, 2).contiguous()  # (B, L, nh, hd)
        out = out.reshape(batch_size, seq_len, self.hidden_dim)  # (B, L, H)

        out = self.out_proj(out)

        return out


# ===== Feed-Forward Networks =====

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    Paper: "Attention is All You Need" (2017)
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1
    ):
        super().__init__()

        # GLU variants use 2/3 expansion (since gating halves dimension)
        if activation in ["geglu", "swiglu"]:
            # Project to 2 * ffn_dim (will be halved by gating)
            self.fc1 = nn.Linear(hidden_dim, 2 * ffn_dim, bias=False)
            self.fc2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        else:
            self.fc1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
            self.fc2 = nn.Linear(ffn_dim, hidden_dim, bias=False)

        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ===== Transformer Block =====

class TransformerBlock(nn.Module):
    """
    Standard Transformer Block

    Architecture:
    - Layer Norm 1
    - Multi-Head Attention
    - Residual connection
    - Layer Norm 2
    - Feed-Forward
    - Residual connection
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        normalization: str = "layernorm",
        activation: str = "gelu",
        attention_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        use_rope: bool = False
    ):
        super().__init__()

        # Pre-norm architecture (LLaMA, GPT-3 style)
        self.norm1 = get_normalization(normalization, hidden_dim)
        self.attn = MultiHeadAttention(
            hidden_dim, num_heads, attention_dropout, use_rope
        )

        self.norm2 = get_normalization(normalization, hidden_dim)
        self.ffn = FeedForward(
            hidden_dim, ffn_dim, activation, residual_dropout
        )

        self.dropout = nn.Dropout(residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Attention block with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask)
        x = self.dropout(x)
        x = residual + x

        # FFN block with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


# ===== Complete Models =====

class TransformerLM(nn.Module):
    """
    Transformer Language Model

    Based on: GPT-2, GPT-3, LLaMA architecture
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()

        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Positional encoding
        self.use_rope = (config.position_encoding == "rope")
        if config.position_encoding == "absolute":
            self.pos_encoding = PositionalEncoding(
                config.hidden_dim, config.max_seq_length
            )
        # RoPE is applied inside attention

        # Transformer blocks
        ffn_dim = int(config.hidden_dim * config.ffn_multiplier)

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                ffn_dim=ffn_dim,
                normalization=config.normalization,
                activation=config.activation,
                attention_dropout=config.attention_dropout,
                residual_dropout=config.residual_dropout,
                use_rope=self.use_rope
            )
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.norm = get_normalization(config.normalization, config.hidden_dim)

        # LM head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying (optional, common in modern LMs)
        # self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights (GPT-2 style)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) - Token IDs
            attention_mask: (batch_size, seq_len) or None - Attention mask
            inputs_embeds: (batch_size, seq_len, hidden_dim) or None - Pre-computed embeddings
            **kwargs: Additional arguments (for PEFT compatibility)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # Handle inputs_embeds (PEFT sometimes passes pre-computed embeddings)
        if inputs_embeds is not None:
            x = inputs_embeds
        elif input_ids is not None:
            # Embed tokens
            x = self.token_embedding(input_ids)  # (B, L, H)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Add positional encoding (if absolute)
        if hasattr(self, 'pos_encoding'):
            x = self.pos_encoding(x)

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final norm
        x = self.norm(x)

        # LM head
        logits = self.lm_head(x)  # (B, L, V)

        return logits

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ):
        """
        Prepare inputs for generation (required by PEFT for CAUSAL_LM).

        This method is used by HuggingFace's generation utilities.
        For LoRA compatibility, we provide a minimal implementation.

        Args:
            input_ids: (batch_size, seq_len)
            **kwargs: Additional arguments (ignored)

        Returns:
            dict with model inputs
        """
        return {"input_ids": input_ids}


# ===== Model Factory =====

def build_model(config: ArchitectureConfig) -> nn.Module:
    """
    Build model from configuration

    Args:
        config: ArchitectureConfig

    Returns:
        nn.Module (TransformerLM, MambaLM, etc.)
    """
    if config.arch_type == "transformer":
        return TransformerLM(config)

    elif config.arch_type == "linear_transformer":
        # TODO: Implement LinearTransformer
        raise NotImplementedError("LinearTransformer not yet implemented")

    elif config.arch_type == "flash_attention":
        # Use TransformerLM with FlashAttention
        # TODO: Integrate flash_attn package
        raise NotImplementedError("FlashAttention not yet implemented")

    elif config.arch_type == "grouped_query_attention":
        # TODO: Implement GQA
        raise NotImplementedError("GQA not yet implemented")

    elif config.arch_type == "mamba":
        # TODO: Implement Mamba
        raise NotImplementedError("Mamba not yet implemented")

    elif config.arch_type == "rwkv":
        # TODO: Implement RWKV
        raise NotImplementedError("RWKV not yet implemented")

    else:
        raise ValueError(f"Unknown architecture type: {config.arch_type}")


if __name__ == "__main__":
    from search_space import get_baseline_architectures

    print("="*60)
    print("Model Architecture Test")
    print("="*60)

    # Test with baseline architectures
    baselines = get_baseline_architectures()

    for i, config in enumerate(baselines[:3], 1):  # Test first 3
        print(f"\n[{i}] Testing {config.arch_type}...")

        try:
            model = build_model(config)

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())

            # Test forward pass
            batch_size = 2
            seq_len = 128
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

            with torch.no_grad():
                logits = model(input_ids)

            print(f"  [OK] Model built successfully")
            print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
            print(f"  Estimated: {config.estimate_parameters():,}")
            print(f"  Output shape: {logits.shape}")
            print(f"  Expected: ({batch_size}, {seq_len}, {config.vocab_size})")

            assert logits.shape == (batch_size, seq_len, config.vocab_size)
            print(f"  [OK] Forward pass successful")

        except NotImplementedError as e:
            print(f"  [SKIP] {e}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            raise

    print(f"\n{'='*60}")
    print("All tests passed!")
    print("="*60)
