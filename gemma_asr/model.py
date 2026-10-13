"""EmbeddingGemma 2 audio encoder (frozen) + a small CTC "reader".

    wav --GPU log-mel--> Gemma audio encoder (USM conformer, 25 frames/s, frozen)
        --> softmax-weighted mix of its 13 hidden states (learned; tells us which layers carry phonetics)
        --> Linear + depthwise conv + N pre-norm transformer layers (the reader)
        --> head over units, rows initialised from Gemma's own token embeddings ("text space")
        --> CTC
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .features import LogMel


@dataclass
class ModelConfig:
    encoder_id: str = "google/embeddinggemma-2"
    vocab_size: int = 0  # number of CTC units incl. blank and unk
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    conv_kernel: int = 15
    dropout: float = 0.1
    layer_mix: bool = True  # learn a weighted sum of all encoder hidden states (else: last layer only)
    unfreeze_top_k: int = 0  # number of top encoder layers to fine-tune (0 = fully frozen encoder)
    head_init: str = "text"  # "text": init unit rows from Gemma token embeddings, "random"
    freeze_head: bool = False  # keep the text-space unit rows fixed (pure "nearest word in text space")
    specaug: dict = field(
        default_factory=lambda: {"freq_masks": 2, "freq_width": 27, "time_masks": 10, "time_width_frac": 0.04}
    )


class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout if self.training else 0.0
        )
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.norm2(x)))


def load_gemma_parts(encoder_id: str, unit_gemma_ids: list[int] | None):
    """Returns (audio_tower, unit token embeddings or None). Drops the rest of EmbeddingGemma 2."""
    full = AutoModel.from_pretrained(encoder_id, dtype=torch.bfloat16, vision_config=None)
    emb = None
    if unit_gemma_ids is not None:
        emb = full.language_model.embed_tokens.weight[torch.tensor(unit_gemma_ids)].detach().float().clone()
    encoder = full.audio_tower
    del full
    return encoder, emb


class GemmaCTC(nn.Module):
    def __init__(self, cfg: ModelConfig, encoder: nn.Module, unit_emb: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        self.mel = LogMel(cfg.encoder_id)
        self.encoder = encoder
        n_enc_layers = encoder.config.num_hidden_layers
        enc_dim = encoder.config.hidden_size

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if cfg.unfreeze_top_k > 0:
            # fp32 master weights for the part we train; the forward pass runs under bf16 autocast
            self.encoder.float()
            for layer in self.encoder.layers[n_enc_layers - cfg.unfreeze_top_k :]:
                for p in layer.parameters():
                    p.requires_grad_(True)

        # Collect the subsampler output + every conformer layer output via hooks. (`output_hidden_states`
        # is not usable: it drops the last layer's raw output in favour of the 1536-d output_proj.)
        self._acts: list[torch.Tensor] = []
        grab = lambda mod, inp, out: self._acts.append(out[0] if isinstance(out, tuple) else out)  # noqa: E731
        self.encoder.subsample_conv_projection.register_forward_hook(grab)
        for layer in self.encoder.layers:
            layer.register_forward_hook(grab)

        d = cfg.d_model
        self.layer_logits = nn.Parameter(torch.zeros(n_enc_layers + 1)) if cfg.layer_mix else None
        self.in_norm = nn.LayerNorm(enc_dim)
        self.in_proj = nn.Linear(enc_dim, d)
        self.pos_conv = nn.Conv1d(d, d, cfg.conv_kernel, padding=cfg.conv_kernel // 2, groups=d)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, cfg.n_heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.out_norm = nn.LayerNorm(d)

        # Head lives in Gemma's text-embedding space (dim 512) so unit rows can be Gemma token embeddings.
        text_dim = 512 if unit_emb is None else unit_emb.shape[1]
        self.to_text = nn.Linear(d, text_dim)
        n_units = cfg.vocab_size - 2
        if cfg.head_init == "text" and unit_emb is not None:
            rows = F.normalize(unit_emb, dim=-1)
        else:
            rows = torch.randn(n_units, text_dim) * text_dim**-0.5
        self.unit_rows = nn.Parameter(rows, requires_grad=not cfg.freeze_head)
        self.special_rows = nn.Parameter(torch.randn(2, text_dim) * text_dim**-0.5)  # blank, unk
        self.head_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    # ---------------------------------------------------------------- encoder
    def _encoder_hidden(self, feats: torch.Tensor, feat_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = torch.no_grad() if self.cfg.unfreeze_top_k == 0 else nullcontext()
        enc_dtype = next(self.encoder.parameters()).dtype
        self._acts.clear()
        with ctx:
            out = self.encoder(feats.to(enc_dtype), feat_mask, return_dict=True)
        acts, self._acts = self._acts, []
        assert len(acts) == self.encoder.config.num_hidden_layers + 1
        if self.cfg.layer_mix:
            w = torch.softmax(self.layer_logits.float(), dim=0)
            h = sum(w[i] * a.float() for i, a in enumerate(acts))
        else:
            h = acts[-1].float()  # last conformer layer
        return h, out.attention_mask

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # no dropout etc. in the pretrained encoder, even when partially fine-tuned
        return self

    def spec_augment(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sa = self.cfg.specaug
        B, T, Fdim = feats.shape
        lens = mask.sum(1)
        fill = (feats * mask[..., None]).sum(1, keepdim=True) / lens.clamp(min=1)[:, None, None]
        drop = torch.zeros(B, T, Fdim, dtype=torch.bool, device=feats.device)
        f_idx = torch.arange(Fdim, device=feats.device)
        t_idx = torch.arange(T, device=feats.device)
        for _ in range(sa["freq_masks"]):
            w = torch.randint(0, sa["freq_width"] + 1, (B, 1), device=feats.device)
            s = (torch.rand(B, 1, device=feats.device) * (Fdim - w)).long()
            drop |= ((f_idx >= s) & (f_idx < s + w))[:, None, :]
        max_w = (lens.float() * sa["time_width_frac"]).long().clamp(min=1)[:, None]
        for _ in range(sa["time_masks"]):
            w = (torch.rand(B, 1, device=feats.device) * (max_w + 1)).long()
            s = (torch.rand(B, 1, device=feats.device) * (lens[:, None] - w).clamp(min=1)).long()
            drop |= ((t_idx >= s) & (t_idx < s + w))[:, :, None]
        return torch.where(drop, fill.expand_as(feats), feats) * mask[..., None]

    # ---------------------------------------------------------------- forward
    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor, specaug: bool = False):
        """Returns CTC logits (B, T, V) in float32 and output lengths (B,)."""
        feats, feat_mask = self.mel(wav, wav_lens)
        if specaug:
            feats = self.spec_augment(feats, feat_mask)
        h, mask = self._encoder_hidden(feats, feat_mask)

        x = self.in_proj(self.in_norm(h))
        x = x + F.gelu(self.pos_conv((x * mask[..., None]).transpose(1, 2)).transpose(1, 2))
        attn_mask = mask[:, None, None, :]  # (B, 1, 1, T) key padding
        for blk in self.blocks:
            x = blk(x, attn_mask)
        t = self.to_text(self.out_norm(x))
        rows = torch.cat([self.special_rows, self.unit_rows], dim=0)
        logits = t.float() @ rows.float().t() + self.head_bias.float()
        return logits, mask.sum(1)

    # ---------------------------------------------------------------- checkpoints
    def trainable_state_dict(self) -> dict:
        """Everything except frozen encoder weights (those come from the HF checkpoint)."""
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.") or k in trainable}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "GemmaCTC":
        cfg = ModelConfig(**ckpt["config"])
        encoder, _ = load_gemma_parts(cfg.encoder_id, None)
        model = cls(cfg, encoder)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        missing = [k for k in missing if not k.startswith("encoder.")]
        if missing or unexpected:
            raise RuntimeError(f"bad checkpoint: missing={missing} unexpected={unexpected}")
        return model

    def config_dict(self) -> dict:
        return asdict(self.cfg)


@torch.no_grad()
def greedy_decode(logits: torch.Tensor, lens: torch.Tensor) -> list[list[int]]:
    """Best-path CTC decoding: argmax, collapse repeats, drop blanks (0)."""
    best = logits.argmax(-1).cpu()
    out = []
    for b in range(best.shape[0]):
        seq = best[b, : int(lens[b])]
        keep = torch.ones_like(seq, dtype=torch.bool)
        keep[1:] = seq[1:] != seq[:-1]
        seq = seq[keep]
        out.append(seq[seq != 0].tolist())
    return out
