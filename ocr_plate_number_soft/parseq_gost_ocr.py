
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Alphabet and subtype-aware grammar
# ============================================================

# One canonical visual alphabet for all countries.
# Russian labels are normalized to their visually equivalent Latin letters
# in data.py, so the decoder never has to learn both "А" and "A" as
# separate tokens for the same glyph.
RUS_PLATE_LETTERS = "ABEKMHOPCTYX"
ALL_PLATE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"


class PlateAlphabet:
    """
    Vocabulary:
      0: <PAD>
      1: <BOS>
      2: <EOS>
      3..: digits + Latin A-Z

    gost_letter_ids contains the 12 visual Latin equivalents allowed on
    ordinary Russian plates.
    """

    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self, letters: str = ALL_PLATE_LETTERS):
        symbols = [self.PAD, self.BOS, self.EOS] + list(DIGITS) + list(letters)
        self.symbols = symbols
        self.stoi = {s: i for i, s in enumerate(symbols)}
        self.itos = {i: s for i, s in enumerate(symbols)}

        self.pad_id = self.stoi[self.PAD]
        self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]

        self.digit_ids = [self.stoi[x] for x in DIGITS]
        self.letter_ids = [self.stoi[x] for x in letters]
        self.gost_letter_ids = [
            self.stoi[x]
            for x in RUS_PLATE_LETTERS
            if x in self.stoi
        ]
        self.char_ids = self.digit_ids + self.letter_ids

    def __len__(self) -> int:
        return len(self.symbols)

    def encode_chars(self, text: str, max_chars: int) -> torch.Tensor:
        out = torch.full((max_chars,), self.pad_id, dtype=torch.long)
        for i, ch in enumerate(text[:max_chars]):
            # '#' is a source-annotation wildcard: the true character is unknown.
            # Represent it as PAD so CrossEntropy(ignore_index=PAD) does not
            # train the decoder to emit a fake '#' class. The position still
            # counts toward the plate length, so EOS is learned at the correct place.
            if ch == "#":
                out[i] = self.pad_id
                continue
            if ch not in self.stoi:
                raise ValueError(f"Unsupported plate symbol: {ch!r}")
            out[i] = self.stoi[ch]
        return out

    def decode_chars(self, ids: Iterable[int]) -> str:
        chars: List[str] = []
        for idx in ids:
            idx = int(idx)
            if idx == self.eos_id:
                break
            if idx in (self.pad_id, self.bos_id):
                continue
            chars.append(self.itos[idx])
        return "".join(chars)


class GostGrammar:
    """
    Subtype-aware constrained decoding.

    Pattern syntax:
      # = digit
      R = one of the 12 Russian/GOST visual letters ABEKMHOPCTYX
      L = any Latin letter A-Z
      X = any alphanumeric symbol A-Z/0-9
      any other character = fixed literal token

    For AM/BY/KG/KZ the supplied CSV files contain several layouts.
    We therefore constrain only the lengths for the heterogeneous datasets
    instead of forcing one national pattern and accidentally making valid
    legacy/special plates impossible to decode.
    """

    def __init__(self, alphabet: PlateAlphabet):
        self.alphabet = alphabet
        self.patterns: Dict[str, List[str]] = {
            # Russian GOST classes
            "type1":  ["R###RR##", "R###RR###"],
            "type1a": ["R###RR##", "R###RR###"],
            "type1b": ["RR#####"],
            "type9":  ["###CD###"],
            "type10": ["###D#####", "###T#####"],

            # Country datasets, based on the uploaded CSV length distributions:
            # AM: 6/7 chars, BY: 7, KG: 5..8, KZ: 6..8.
            "AM": ["XXXXXX", "XXXXXXX"],
            "BY": ["XXXXXXX"],
            "KG": ["XXXXX", "XXXXXX", "XXXXXXX", "XXXXXXXX"],
            "KZ": ["XXXXXX", "XXXXXXX", "XXXXXXXX"],
        }

    def register(self, subtype: str, *patterns: str) -> None:
        for pattern in patterns:
            if not pattern:
                raise ValueError("Empty OCR grammar pattern")
            for ch in pattern:
                if ch in {"#", "R", "L", "X"}:
                    continue
                if ch not in self.alphabet.stoi:
                    raise ValueError(
                        f"Unsupported literal {ch!r} in pattern {pattern!r}"
                    )
        self.patterns[subtype] = list(patterns)

    def allowed_token_ids(self, subtype: str, char_pos: int) -> List[int]:
        patterns = self.patterns.get(subtype)

        # Unregistered subtype: fully unconstrained alphanumeric decoding.
        if not patterns:
            ids = list(self.alphabet.char_ids)
            if char_pos > 0:
                ids.append(self.alphabet.eos_id)
            return ids

        allowed: set[int] = set()

        # EOS is legal when at least one valid pattern ends here.
        if any(len(p) == char_pos for p in patterns):
            allowed.add(self.alphabet.eos_id)

        symbols = {p[char_pos] for p in patterns if char_pos < len(p)}
        for symbol in symbols:
            if symbol == "#":
                allowed.update(self.alphabet.digit_ids)
            elif symbol == "R":
                allowed.update(self.alphabet.gost_letter_ids)
            elif symbol == "L":
                allowed.update(self.alphabet.letter_ids)
            elif symbol == "X":
                allowed.update(self.alphabet.char_ids)
            else:
                allowed.add(self.alphabet.stoi[symbol])

        return sorted(allowed)

# ============================================================
# 2. 2D RoPE visual encoder
# ============================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class Rotary1D(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE dimension must be even.")
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.dim = dim

    def apply(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, H, N, D]
        pos: [N] or [B, N]
        """
        if pos.dim() == 1:
            angles = pos.float()[:, None] * self.inv_freq[None, :]
            cos = angles.cos().repeat_interleave(2, dim=-1)[None, None, :, :]
            sin = angles.sin().repeat_interleave(2, dim=-1)[None, None, :, :]
        else:
            angles = pos.float()[..., None] * self.inv_freq[None, None, :]
            cos = angles.cos().repeat_interleave(2, dim=-1)[:, None, :, :]
            sin = angles.sin().repeat_interleave(2, dim=-1)[:, None, :, :]
        return x * cos.to(x.dtype) + _rotate_half(x) * sin.to(x.dtype)


class Rotary2D(nn.Module):
    """
    Splits each attention head into X and Y halves and applies separate 1D RoPE.
    head_dim must be divisible by 4.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 4:
            raise ValueError("For 2D RoPE, head_dim must be divisible by 4.")
        axis_dim = head_dim // 2
        self.rope_x = Rotary1D(axis_dim, base)
        self.rope_y = Rotary1D(axis_dim, base)
        self.axis_dim = axis_dim

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coords_xy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: [B, heads, N, head_dim]
        coords_xy: [N, 2], columns = x, y
        """
        x = coords_xy[:, 0]
        y = coords_xy[:, 1]

        qx, qy = q[..., :self.axis_dim], q[..., self.axis_dim:]
        kx, ky = k[..., :self.axis_dim], k[..., self.axis_dim:]

        qx = self.rope_x.apply(qx, x)
        kx = self.rope_x.apply(kx, x)
        qy = self.rope_y.apply(qy, y)
        ky = self.rope_y.apply(ky, y)

        return torch.cat([qx, qy], dim=-1), torch.cat([kx, ky], dim=-1)


class PatchStem(nn.Module):
    """
    Lightweight CNN stem with total stride 4.
    Keeps a genuine 2D feature map before flattening to visual tokens.
    """

    def __init__(self, in_ch: int, dim: int):
        super().__init__()
        c1 = max(32, dim // 4)
        c2 = max(64, dim // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, dim, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.SiLU(inplace=True),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fmap = self.net(x)                       # [B, D, H', W']
        b, d, h, w = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)  # [B, N, D]
        tokens = self.norm(tokens)

        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device),
            torch.arange(w, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1).float()
        return tokens, coords


class RoPE2DSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 4:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
        self.rope = Rotary2D(self.head_dim)

    def forward(self, x: torch.Tensor, coords_xy: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B,H,N,Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rope.apply(q, k, coords_xy)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(F.silu(a) * b))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = RoPE2DSelfAttention(dim, heads, dropout)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, int(dim * mlp_ratio), dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.n1(x), coords))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x


# ============================================================
# 3. PARSeq-style two-stream decoder
# ============================================================

def _expand_attn_mask(mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    nn.MultiheadAttention accepts [L,S] or [B*H,L,S].
    Input mask here is bool [B,L,S], where True means "forbidden".
    """
    if mask.dim() == 2:
        return mask
    b, l, s = mask.shape
    return (
        mask[:, None, :, :]
        .expand(b, num_heads, l, s)
        .reshape(b * num_heads, l, s)
    )


class PARSeqDecoderLayer(nn.Module):
    """
    Compact two-stream decoder:
      - content stream contains BOS + ground-truth / generated characters
      - query stream contains learned positional queries
      - query stream attends to content under a permutation/causal mask
      - both streams cross-attend to the visual encoder memory

    This preserves the main PARSeq idea without copying the official code.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.heads = heads

        self.c_n1 = nn.LayerNorm(dim)
        self.q_n1 = nn.LayerNorm(dim)
        self.content_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.query_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )

        self.c_n2 = nn.LayerNorm(dim)
        self.q_n2 = nn.LayerNorm(dim)
        self.content_cross = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.query_cross = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )

        hidden = int(dim * mlp_ratio)
        self.c_n3 = nn.LayerNorm(dim)
        self.q_n3 = nn.LayerNorm(dim)
        self.c_ffn = SwiGLU(dim, hidden, dropout)
        self.q_ffn = SwiGLU(dim, hidden, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        content: torch.Tensor,
        query: torch.Tensor,
        memory: torch.Tensor,
        content_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cm = _expand_attn_mask(content_mask, self.heads)
        qm = _expand_attn_mask(query_mask, self.heads)

        c0 = self.c_n1(content)
        c_attn, _ = self.content_attn(
            c0, c0, c0, attn_mask=cm, need_weights=False
        )
        content = content + self.drop(c_attn)

        q0 = self.q_n1(query)
        # Query stream reads the content stream but is prevented from
        # seeing the current/future token by the permutation mask.
        q_attn, _ = self.query_attn(
            q0, self.c_n1(content), self.c_n1(content),
            attn_mask=qm, need_weights=False
        )
        query = query + self.drop(q_attn)

        c1 = self.c_n2(content)
        c_cross, _ = self.content_cross(c1, memory, memory, need_weights=False)
        content = content + self.drop(c_cross)

        q1 = self.q_n2(query)
        q_cross, _ = self.query_cross(q1, memory, memory, need_weights=False)
        query = query + self.drop(q_cross)

        content = content + self.drop(self.c_ffn(self.c_n3(content)))
        query = query + self.drop(self.q_ffn(self.q_n3(query)))
        return content, query


class PARSeqDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        heads: int,
        depth: int,
        max_chars: int,
        pad_id: int,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.max_chars = max_chars
        self.seq_len = max_chars + 1  # chars + EOS
        self.pad_id = pad_id

        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.content_pos = nn.Parameter(torch.randn(1, self.seq_len, dim) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, self.seq_len, dim) * 0.02)

        self.layers = nn.ModuleList([
            PARSeqDecoderLayer(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(
        self,
        content_ids: torch.Tensor,
        memory: torch.Tensor,
        content_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, t = content_ids.shape
        if t != self.seq_len:
            raise ValueError(f"Expected decoder sequence length {self.seq_len}, got {t}")

        content = self.token_embed(content_ids) + self.content_pos[:, :t]
        query = self.query_pos[:, :t].expand(b, -1, -1)

        for layer in self.layers:
            content, query = layer(
                content, query, memory, content_mask, query_mask
            )

        return self.head(self.norm(query))


# ============================================================
# 4. Mask generation for PARSeq training and LTR inference
# ============================================================

def make_ltr_masks(
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    content index 0 = BOS
    content index 1 = char_0
    query index 0 predicts char_0

    Query t may see content positions <= t:
      t=0 -> BOS
      t=1 -> BOS + char_0
      ...
    """
    upper = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    cm = upper[None].expand(batch_size, -1, -1).clone()
    qm = upper[None].expand(batch_size, -1, -1).clone()
    return cm, qm


def make_permutation_masks(
    lengths: torch.Tensor,
    max_chars: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates one random permutation per sample.

    targets:
      positions 0..length-1 = characters
      position length       = EOS

    content:
      position 0   = BOS
      position j+1 = character j
    """
    b = int(lengths.numel())
    t = max_chars + 1

    cm = torch.ones((b, t, t), dtype=torch.bool, device=device)
    qm = torch.ones((b, t, t), dtype=torch.bool, device=device)

    for bi in range(b):
        n = int(lengths[bi].item())
        perm = torch.randperm(n, device=device) if n > 0 else torch.empty(0, dtype=torch.long, device=device)

        # rank[j] = order in which character position j becomes visible
        rank = torch.empty(n, dtype=torch.long, device=device)
        if n > 0:
            rank[perm] = torch.arange(n, device=device)

        # BOS content row: only BOS
        cm[bi, 0, 0] = False

        # Each content char sees BOS + chars not later than itself in the permutation.
        for j in range(n):
            row = j + 1
            cm[bi, row, 0] = False
            visible = torch.nonzero(rank <= rank[j], as_tuple=False).flatten()
            if visible.numel():
                cm[bi, row, visible + 1] = False

        # Query for character j sees BOS + chars earlier in the permutation.
        for j in range(n):
            qm[bi, j, 0] = False
            visible = torch.nonzero(rank < rank[j], as_tuple=False).flatten()
            if visible.numel():
                qm[bi, j, visible + 1] = False

        # EOS query sees all actual characters.
        eos_pos = n
        qm[bi, eos_pos, 0] = False
        if n > 0:
            qm[bi, eos_pos, 1:n + 1] = False

        # Padded rows: let them see BOS to avoid all-masked attention.
        for row in range(n + 1, t):
            cm[bi, row, 0] = False
            qm[bi, row, 0] = False

    return cm, qm


# ============================================================
# 5. Full model: 2D RoPE + subtype + PARSeq + GOST constraints
# ============================================================

@dataclass
class ModelConfig:
    in_ch: int = 3
    dim: int = 256
    encoder_depth: int = 6
    decoder_depth: int = 2
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_register_tokens: int = 4
    max_chars: int = 9
    subtypes: Tuple[str, ...] = ("type1", "type1a", "type1b", "type9", "type10", "AM", "BY", "KG", "KZ")
    type_loss_weight: float = 0.25


class PARSeqGostOCR(nn.Module):
    def __init__(
        self,
        config: ModelConfig = ModelConfig(),
        alphabet: Optional[PlateAlphabet] = None,
    ):
        super().__init__()
        self.cfg = config
        self.alphabet = alphabet or PlateAlphabet()
        self.grammar = GostGrammar(self.alphabet)

        if config.dim % config.heads:
            raise ValueError("dim must be divisible by heads")
        if (config.dim // config.heads) % 4:
            raise ValueError("(dim / heads) must be divisible by 4 for 2D RoPE")

        self.stem = PatchStem(config.in_ch, config.dim)
        self.register_tokens = nn.Parameter(
            torch.randn(1, config.num_register_tokens, config.dim) * 0.02
        )

        self.encoder = nn.ModuleList([
            EncoderBlock(
                config.dim, config.heads,
                config.mlp_ratio, config.dropout
            )
            for _ in range(config.encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(config.dim)

        self.subtype_norm = nn.LayerNorm(config.dim)
        self.subtype_head = nn.Linear(config.dim, len(config.subtypes))

        self.decoder = PARSeqDecoder(
            vocab_size=len(self.alphabet),
            dim=config.dim,
            heads=config.heads,
            depth=config.decoder_depth,
            max_chars=config.max_chars,
            pad_id=self.alphabet.pad_id,
            dropout=config.dropout,
            mlp_ratio=config.mlp_ratio,
        )

    @property
    def subtype_to_id(self) -> Dict[str, int]:
        return {s: i for i, s in enumerate(self.cfg.subtypes)}

    @property
    def id_to_subtype(self) -> Dict[int, str]:
        return {i: s for i, s in enumerate(self.cfg.subtypes)}

    def encode_image(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        images: [B,3,H,W], preferably aspect-ratio-preserving resize/pad.
        Returns:
          visual memory: [B,N,D] (without register tokens)
          subtype logits: [B,num_subtypes]
        """
        tokens, coords = self.stem(images)
        b = tokens.shape[0]

        regs = self.register_tokens.expand(b, -1, -1)
        x = torch.cat([regs, tokens], dim=1)

        # Register tokens receive zero coordinates => identity RoPE rotation.
        reg_coords = torch.zeros(
            (self.cfg.num_register_tokens, 2),
            dtype=coords.dtype,
            device=coords.device,
        )
        all_coords = torch.cat([reg_coords, coords], dim=0)

        for block in self.encoder:
            x = block(x, all_coords)
        x = self.encoder_norm(x)

        reg_out = x[:, :self.cfg.num_register_tokens]
        memory = x[:, self.cfg.num_register_tokens:]

        summary = self.subtype_norm(reg_out.mean(dim=1))
        subtype_logits = self.subtype_head(summary)
        return memory, subtype_logits

    def build_training_tensors(
        self,
        texts: Sequence[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          chars:      [B,max_chars]  padded character ids
          targets:    [B,max_chars+1] chars + EOS + PAD
          lengths:    [B] actual number of characters
        """
        b = len(texts)
        m = self.cfg.max_chars
        chars = torch.full(
            (b, m), self.alphabet.pad_id,
            dtype=torch.long, device=device
        )
        targets = torch.full(
            (b, m + 1), self.alphabet.pad_id,
            dtype=torch.long, device=device
        )
        lengths = torch.zeros((b,), dtype=torch.long, device=device)

        for i, text in enumerate(texts):
            if len(text) > m:
                raise ValueError(f"Plate {text!r} exceeds max_chars={m}")
            encoded = self.alphabet.encode_chars(text, m).to(device)
            chars[i] = encoded
            n = len(text)
            lengths[i] = n
            if n:
                targets[i, :n] = encoded[:n]
            targets[i, n] = self.alphabet.eos_id

        return chars, targets, lengths

    def _make_content_ids(self, chars: torch.Tensor) -> torch.Tensor:
        b, m = chars.shape
        content = torch.full(
            (b, m + 1), self.alphabet.pad_id,
            dtype=torch.long, device=chars.device
        )
        content[:, 0] = self.alphabet.bos_id
        content[:, 1:] = chars
        return content

    def compute_loss(
        self,
        images: torch.Tensor,
        texts: Sequence[str],
        subtype_labels: Sequence[str],
        num_permutations: int = 6,
    ) -> Dict[str, torch.Tensor]:
        """
        One training step loss.

        The image encoder is executed once. The decoder is then trained under
        multiple random target permutations, as in PARSeq-style PLM training.
        """
        device = images.device
        memory, subtype_logits = self.encode_image(images)

        chars, targets, lengths = self.build_training_tensors(texts, device)
        content_ids = self._make_content_ids(chars)

        subtype_map = self.subtype_to_id
        unknown_labels = sorted({s for s in subtype_labels if s not in subtype_map})
        if unknown_labels:
            raise ValueError(
                f"Subtype labels not present in model config: {unknown_labels}. "
                f"Configured subtypes: {list(self.cfg.subtypes)}"
            )
        subtype_ids = torch.tensor(
            [subtype_map[s] for s in subtype_labels],
            dtype=torch.long, device=device
        )

        type_loss = F.cross_entropy(subtype_logits, subtype_ids)

        text_losses = []
        perms = max(1, int(num_permutations))
        for p in range(perms):
            # First pass is always left-to-right; the others are random.
            if p == 0:
                cm, qm = make_ltr_masks(
                    images.shape[0], self.cfg.max_chars + 1, device
                )
            else:
                cm, qm = make_permutation_masks(
                    lengths, self.cfg.max_chars, device
                )

            logits = self.decoder(content_ids, memory, cm, qm)
            text_loss = F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                ignore_index=self.alphabet.pad_id,
            )
            text_losses.append(text_loss)

        text_loss = torch.stack(text_losses).mean()
        loss = text_loss + self.cfg.type_loss_weight * type_loss

        return {
            "loss": loss,
            "text_loss": text_loss.detach(),
            "type_loss": type_loss.detach(),
        }

    def _mask_logits_by_gost(
        self,
        logits: torch.Tensor,
        subtype: str,
        char_pos: int,
    ) -> torch.Tensor:
        """
        logits: [V]
        """
        allowed = self.grammar.allowed_token_ids(subtype, char_pos)
        if not allowed:
            return logits

        mask = torch.full_like(logits, float("-inf"))
        idx = torch.tensor(allowed, device=logits.device, dtype=torch.long)
        mask[idx] = 0.0
        return logits + mask

    @torch.no_grad()
    def recognize(
        self,
        images: torch.Tensor,
        force_subtypes: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        """
        Greedy left-to-right decoding with GOST-constrained token masking.

        Returns list of:
          {
            "text": ...,
            "subtype": ...,
            "subtype_confidence": ...,
            "char_confidence": ...
          }
        """
        self.eval()
        device = images.device
        memory, subtype_logits = self.encode_image(images)
        subtype_probs = subtype_logits.softmax(dim=-1)

        b = images.shape[0]
        results = []

        for bi in range(b):
            if force_subtypes is not None:
                subtype = force_subtypes[bi]
                subtype_conf = 1.0
            else:
                sid = int(subtype_probs[bi].argmax().item())
                subtype = self.id_to_subtype[sid]
                subtype_conf = float(subtype_probs[bi, sid].item())

            chars = torch.full(
                (1, self.cfg.max_chars),
                self.alphabet.pad_id,
                dtype=torch.long,
                device=device,
            )
            char_ids: List[int] = []
            char_probs: List[float] = []

            for pos in range(self.cfg.max_chars + 1):
                content_ids = self._make_content_ids(chars)
                cm, qm = make_ltr_masks(
                    1, self.cfg.max_chars + 1, device
                )
                logits = self.decoder(
                    content_ids,
                    memory[bi:bi + 1],
                    cm,
                    qm,
                )[0, pos]

                constrained = self._mask_logits_by_gost(
                    logits, subtype, pos
                )
                probs = constrained.softmax(dim=-1)
                token_id = int(probs.argmax().item())
                prob = float(probs[token_id].item())

                if token_id == self.alphabet.eos_id:
                    break

                # Safety: grammar should prevent these, but do not insert specials.
                if token_id in (self.alphabet.pad_id, self.alphabet.bos_id):
                    break

                char_ids.append(token_id)
                char_probs.append(prob)

                if pos < self.cfg.max_chars:
                    chars[0, pos] = token_id

            text = self.alphabet.decode_chars(char_ids)
            results.append({
                "text": text,
                "subtype": subtype,
                "subtype_confidence": subtype_conf,
                "char_confidence": (
                    float(sum(char_probs) / len(char_probs))
                    if char_probs else 0.0
                ),
            })

        return results


# ============================================================
# 6. Minimal smoke test
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(7)

    cfg = ModelConfig(
        dim=128,
        encoder_depth=2,
        decoder_depth=2,
        heads=4,
        max_chars=9,
    )
    model = PARSeqGostOCR(cfg)

    # IMPORTANT:
    # Keep plate aspect ratio in the real pipeline.
    # 64x128 is used here only for a small smoke test.
    x = torch.randn(2, 3, 64, 128)

    losses = model.compute_loss(
        x,
        texts=["А123ВС77", "Т706НХ799"],
        subtype_labels=["type1", "type1a"],
        num_permutations=2,
    )
    print({k: float(v) for k, v in losses.items()})

    out = model.recognize(
        x[:1],
        force_subtypes=["type1a"],
    )
    print(out)
