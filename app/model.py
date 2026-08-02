"""
Model definition for the multimodal dermoscopy classifier.

Extracted verbatim (behaviour-preserving) from the thesis notebook so that
`load_state_dict` on the trained checkpoint matches exactly. The champion is the
one-way variant (`adapter_type='oneway'`, `adapter_layers=2`,
`use_derm_token=False`), but both adapter classes are kept so the module is
self-contained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm


class StableProjectedCrossAttnAdapter(nn.Module):
    """Bidirectional cross-attention adapter (NOT used by the champion).

    Kept for completeness / reproducing the bidirectional ablation. The
    duplicated-init present in the notebook has been removed; this does not
    affect the one-way champion, which never instantiates this class.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1, ffn_expansion=2):
        super().__init__()
        ffn_hidden = ffn_expansion * embed_dim

        # Image pathway
        self.img_norm_q = nn.LayerNorm(embed_dim)
        self.meta_norm_kv_for_img = nn.LayerNorm(embed_dim)
        self.img_q_proj = nn.Linear(embed_dim, embed_dim)
        self.img_k_proj = nn.Linear(embed_dim, embed_dim)
        self.img_v_proj = nn.Linear(embed_dim, embed_dim)
        self.img_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.img_norm_ffn = nn.LayerNorm(embed_dim)
        self.img_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_hidden, embed_dim), nn.Dropout(dropout),
        )
        self.gate_attn = nn.Parameter(torch.full((1,), 1e-2))
        self.gate_ffn = nn.Parameter(torch.full((1,), 1e-2))

        # Metadata pathway
        self.meta_norm_q = nn.LayerNorm(embed_dim)
        self.img_norm_kv_for_meta = nn.LayerNorm(embed_dim)
        self.meta_q_proj = nn.Linear(embed_dim, embed_dim)
        self.meta_k_proj = nn.Linear(embed_dim, embed_dim)
        self.meta_v_proj = nn.Linear(embed_dim, embed_dim)
        self.meta_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.meta_norm_ffn = nn.LayerNorm(embed_dim)
        self.meta_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_hidden, embed_dim), nn.Dropout(dropout),
        )

    def set_gates(self, attn_value, ffn_value, trainable):
        with torch.no_grad():
            self.gate_attn.fill_(attn_value)
            self.gate_ffn.fill_(ffn_value)
        self.gate_attn.requires_grad_(trainable)
        self.gate_ffn.requires_grad_(trainable)

    def forward(self, img_tokens, meta_tokens):
        # 1. Image attends to metadata
        q = self.img_q_proj(self.img_norm_q(img_tokens))
        norm_meta_kv = self.meta_norm_kv_for_img(meta_tokens)
        k = self.img_k_proj(norm_meta_kv)
        v = self.img_v_proj(norm_meta_kv)
        img_attn_out, _ = self.img_attn(q, k, v)
        img_tokens = img_tokens + (self.gate_attn * img_attn_out)
        img_tokens = img_tokens + (self.gate_ffn * self.img_ffn(self.img_norm_ffn(img_tokens)))

        # 2. Metadata attends to image
        q = self.meta_q_proj(self.meta_norm_q(meta_tokens))
        norm_img_kv = self.img_norm_kv_for_meta(img_tokens)
        k = self.meta_k_proj(norm_img_kv)
        v = self.meta_v_proj(norm_img_kv)
        meta_attn_out, _ = self.meta_attn(q, k, v)
        meta_tokens = meta_tokens + meta_attn_out
        meta_tokens = meta_tokens + self.meta_ffn(self.meta_norm_ffn(meta_tokens))
        return img_tokens, meta_tokens


class MetaFromImgCrossAttnAdapter(nn.Module):
    """One-way adapter: metadata queries the image stream, image stream is
    read-only. This is the champion adapter."""

    def __init__(self, embed_dim, num_heads, dropout=0.1, ffn_expansion=2):
        super().__init__()
        ffn_hidden = ffn_expansion * embed_dim

        self.meta_norm_q = nn.LayerNorm(embed_dim)
        self.img_norm_kv_for_meta = nn.LayerNorm(embed_dim)
        self.meta_q_proj = nn.Linear(embed_dim, embed_dim)
        self.meta_k_proj = nn.Linear(embed_dim, embed_dim)
        self.meta_v_proj = nn.Linear(embed_dim, embed_dim)
        self.meta_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.meta_norm_ffn = nn.LayerNorm(embed_dim)
        self.meta_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_hidden, embed_dim), nn.Dropout(dropout),
        )

    def set_gates(self, attn_value, ffn_value, trainable):
        """No-op for the one-way adapter — kept for API compatibility."""
        pass

    def forward(self, img_tokens, meta_tokens):
        q = self.meta_q_proj(self.meta_norm_q(meta_tokens))
        norm_img_kv = self.img_norm_kv_for_meta(img_tokens)
        k = self.meta_k_proj(norm_img_kv)
        v = self.meta_v_proj(norm_img_kv)
        meta_attn_out, _ = self.meta_attn(q, k, v)
        meta_tokens = meta_tokens + meta_attn_out
        meta_tokens = meta_tokens + self.meta_ffn(self.meta_norm_ffn(meta_tokens))
        return img_tokens, meta_tokens


_ADAPTERS = {
    "bidirectional": StableProjectedCrossAttnAdapter,
    "oneway": MetaFromImgCrossAttnAdapter,
}


class ViT_HAM10000_HybridCNN(nn.Module):
    """Hybrid CNN-ViT (vit_base_r50_s16_224) with cross-attention adapters that
    fuse image patch tokens and metadata tokens."""

    def __init__(
        self,
        vit_name="vit_base_r50_s16_224",
        num_num=4,
        cat_cardinalities=None,
        num_classes=7,
        adapter_layers=2,
        num_heads=8,
        dropout=0.5,
        use_derm_token=False,
        derm_dim=6144,
        unfreeze_last_n_blocks=2,
        adapter_type="oneway",
        unfreeze_cnn_stem=False,
        pretrained=False,  # inference: weights come from the checkpoint, not timm
    ):
        super().__init__()
        # For inference we do not need timm's pretrained download; the trained
        # checkpoint overwrites every weight anyway. Set pretrained=True only if
        # you are re-instantiating for training.
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0, drop_path_rate=0.2)
        self.embed_dim = self.vit.embed_dim  # 768

        self.num_num = num_num
        self.cat_cardinalities = cat_cardinalities or []

        self.num_token_proj = nn.ModuleList([nn.Linear(1, self.embed_dim) for _ in range(num_num)])
        self.cat_token_emb = nn.ModuleList([nn.Embedding(card, self.embed_dim) for card in self.cat_cardinalities])

        self.use_derm_token = use_derm_token
        if self.use_derm_token:
            self.derm_token_proj = nn.Sequential(nn.LayerNorm(derm_dim), nn.Linear(derm_dim, self.embed_dim))

        AdapterClass = _ADAPTERS[adapter_type]
        self.adapter_type = adapter_type
        self.adapter_layers = adapter_layers
        self.adapters = nn.ModuleList([
            AdapterClass(self.embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(adapter_layers)
        ])

        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.embed_dim * 2, num_classes))

        # Freezing is a training concern; irrelevant at inference (model.eval()).
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.unfreeze_cnn_stem = unfreeze_cnn_stem

    def build_meta_tokens(self, x_num, x_cat, x_derm=None):
        token_list = []
        if self.num_num > 0:
            toks = [proj(x_num[:, i:i + 1]).unsqueeze(1) for i, proj in enumerate(self.num_token_proj)]
            token_list.append(torch.cat(toks, dim=1))
        if len(self.cat_cardinalities) > 0:
            toks = [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_token_emb)]
            token_list.append(torch.cat(toks, dim=1))
        if self.use_derm_token and x_derm is not None:
            token_list.append(self.derm_token_proj(x_derm).unsqueeze(1))
        return torch.cat(token_list, dim=1)

    def forward(self, x_img, x_num, x_cat, x_derm=None):
        B = x_img.size(0)
        x = self.vit.patch_embed(x_img)
        cls = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        meta_kv = self.build_meta_tokens(x_num, x_cat, x_derm)

        n_blocks = len(self.vit.blocks)
        start = n_blocks - self.adapter_layers
        adapter_idx = 0
        for block_idx, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if block_idx >= start:
                x, meta_kv = self.adapters[adapter_idx](x, meta_kv)
                adapter_idx += 1

        x = self.vit.norm(x)
        img_cls = x[:, 0, :]
        meta_cls = meta_kv.mean(dim=1)
        return self.head(torch.cat([img_cls, meta_cls], dim=1))

    @torch.no_grad()
    def embed_image(self, x_img):
        """Image-only CLS embedding, for out-of-distribution checks.

        Safe to compute without any metadata: the "oneway" adapter
        (`MetaFromImgCrossAttnAdapter`) only ever updates `meta_tokens` and
        returns `img_tokens` unchanged, so `img_cls` in `forward()` never
        depends on x_num/x_cat/x_derm for this champion. This method
        reproduces exactly that image-token pathway, skipping the
        metadata-side computation entirely.
        """
        B = x_img.size(0)
        x = self.vit.patch_embed(x_img)
        cls = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        return x[:, 0, :]


# Champion config, kept next to the model so the loader and export stay in sync.
CHAMPION_CONFIG = dict(
    vit_name="vit_base_r50_s16_224",
    num_classes=7,
    adapter_layers=2,
    num_heads=8,
    dropout=0.5,
    use_derm_token=False,
    unfreeze_last_n_blocks=2,
    adapter_type="oneway",
    unfreeze_cnn_stem=False,
)

CLASS_NAMES = ["NV", "MEL", "BCC", "AKIEC", "BKL", "DF", "VASC"]
# Indices treated as malignant/pre-malignant for the aggregate sensitivity view.
MALIGNANT_IDX = [1, 2, 3]  # MEL, BCC, AKIEC
