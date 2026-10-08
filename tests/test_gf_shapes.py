"""CPU shape tests with a fake tiny LLM (no GPU, no download). Put in tests/ ; run: python -m pytest tests -v"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "gf"))
import pandas as pd  # noqa: E402
from gf_data import augment_clip, make_split  # noqa: E402
from gf_model import SignLLM, encode_targets  # noqa: E402


class FakeTok:
    bos_token_id, eos_token_id = 1, 2
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + (ord(c) % 40) for c in text]}


class FakeLLM(nn.Module):
    def __init__(self, V=50, D=32):
        super().__init__()
        self.emb, self.head, self.V = nn.Embedding(V, D), nn.Linear(D, V), V
    def get_input_embeddings(self): return self.emb
    def forward(self, inputs_embeds=None, attention_mask=None, labels=None):
        L = inputs_embeds.size(1)
        # cumulative mean = tiny "attention": every position can see the earlier ones
        counts = torch.arange(1, L + 1, dtype=inputs_embeds.dtype, device=inputs_embeds.device).view(1, -1, 1)
        logits = self.head(torch.cumsum(inputs_embeds, dim=1) / counts)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.V), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)


def make():
    return SignLLM(FakeLLM(), FakeTok(), "si", n_tokens=16, enc_dim=256, enc_layers=3)


def test_encoder_projector_shapes():
    sl = make()
    outs = sl.encoder(torch.randn(2, 64, 393))
    assert len(outs) == 3 and outs[0].shape == (2, 64, 256)
    assert sl.projector(outs).shape == (2, 16, 32)


def test_loss_and_gradients():
    sl = make()
    t, m = encode_targets(sl.tok, ["abc de", "x"], sl.eos, "cpu")
    loss = sl(torch.randn(2, 64, 393), t, m)
    assert torch.isfinite(loss)
    loss.backward()
    assert sl.projector.mlp[0].weight.grad.abs().sum() > 0
    assert sl.encoder.inp[0].weight.grad.abs().sum() > 0     # gradient reaches the encoder too


def test_confidence_range():
    sl = make()
    t, m = encode_targets(sl.tok, ["abc de", "x"], sl.eos, "cpu")
    c = sl.confidence(torch.randn(2, 64, 393), t, m)
    assert c.shape == (2,) and ((c > 0) & (c <= 1)).all()


def test_augment_keeps_shape_and_zeros():
    rng = np.random.default_rng(0)
    f = np.random.randn(64, 390).astype("float32")
    f[:, 90:120] = 0
    m = np.ones((64, 3), bool)
    f2, m2 = augment_clip(f, m, rng)
    assert f2.shape == (64, 390) and m2.shape == (64, 3)
    assert np.all(f2[:, 90:120] == 0)


def test_split_holds_out_last_take():
    clips = pd.DataFrame([{"stem": "001_t1", "id": 1, "take": 1}, {"stem": "001_t2", "id": 1, "take": 2},
                          {"stem": "001_t3", "id": 1, "take": 3}, {"stem": "002_t1", "id": 2, "take": 1}])
    tr, te = make_split(clips, {1, 2}, save=False)
    assert te == ["001_t3"] and set(tr) == {"001_t1", "001_t2", "002_t1"}
