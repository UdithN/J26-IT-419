"""Gloss-free model: keypoint encoder -> layer-fusion projector -> visual tokens -> LLM.
Sinhala: SinLlama (PEFT adapter on Llama-3-8B).  English: Llama-3-8B + fresh LoRA.
Put in src/gf/."""
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

BASE = "meta-llama/Meta-Llama-3-8B"
ADAPTER = "polyglots/SinLlama_v01"
PROMPT = {"si": "Sinhala sentence:", "en": "English sentence:"}


# ------------------------------------------------------------------ small networks
class SignEncoder(nn.Module):
    """Transformer over 64 frames of (390 keypoint values + 3 mask values). Returns ALL layer outputs."""

    def __init__(self, in_dim=393, d=256, layers=3, heads=4, drop=0.2, n_frames=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.Dropout(drop))
        self.pos = nn.Parameter(torch.randn(1, n_frames, d) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=drop,
                                       batch_first=True, norm_first=True, activation="gelu")
            for _ in range(layers)])

    def forward(self, x):                       # (B,64,393)
        h = self.inp(x) + self.pos
        outs = []
        for layer in self.layers:
            h = layer(h)
            outs.append(h)
        return outs                             # list of (B,64,d)


class LayerFusionProjector(nn.Module):
    """Fuse several encoder layers -> pool 64 frames to K tokens -> map into the LLM embedding space."""

    def __init__(self, d=256, n_layers=3, llm_dim=4096, n_tokens=16, hidden=512, init_scale=0.02):
        super().__init__()
        self.fuse = nn.Sequential(nn.Linear(d * n_layers, hidden), nn.GELU())
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.mlp = nn.Sequential(nn.Linear(hidden, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim))
        self.norm = nn.LayerNorm(llm_dim)
        # real word embeddings are tiny; start the visual tokens at the same scale so they do not wreck the LLM
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, outs):
        h = self.fuse(torch.cat(outs, dim=-1))                  # (B,64,hidden)
        h = self.pool(h.transpose(1, 2)).transpose(1, 2)        # (B,K,hidden)
        return self.norm(self.mlp(h)) * self.scale              # (B,K,llm_dim)


# ------------------------------------------------------------------ LLM loading (validated on Colab T4)
def _copy_sinllama_embeddings(base):
    """Copy SinLlama's trained embed_tokens / lm_head from the adapter file (prefer the modules_to_save copy)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(ADAPTER, "adapter_model.safetensors")
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for name, getter in (("embed_tokens", base.get_input_embeddings), ("lm_head", base.get_output_embeddings)):
            cand = [k for k in keys if name in k and "modules_to_save" in k] or \
                   [k for k in keys if name in k and "lora" not in k]
            w = f.get_tensor(cand[0])
            tgt = getter().weight
            assert tgt.shape == w.shape, (name, tuple(tgt.shape), tuple(w.shape))
            tgt.data.copy_(w.to(tgt.dtype))
            print("copied", cand[0], tuple(w.shape))
            del w


def load_llm(lang: str):
    """Returns (tokenizer, llm (PEFT model, everything frozen), base)."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(ADAPTER if lang == "si" else BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0},
                                                dtype=torch.float16)
    if lang == "si":
        base.resize_token_embeddings(len(tok), mean_resizing=False)
        _copy_sinllama_embeddings(base)
        cfg = PeftConfig.from_pretrained(ADAPTER)
        cfg.modules_to_save = None                       # embeddings already copied: no duplicate copies on GPU
        llm = PeftModel.from_pretrained(base, ADAPTER, config=cfg, torch_device="cpu")
    else:
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
                          task_type="CAUSAL_LM")
        llm = get_peft_model(base, lcfg)
    for p in llm.parameters():
        p.requires_grad = False
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, llm, base


def set_trainable_lora(llm, lang: str) -> int:
    """Phase 2: unfreeze LoRA (Sinhala: only q_proj/v_proj of SinLlama's adapter; English: the fresh LoRA)."""
    n = 0
    for name, p in llm.named_parameters():
        if "lora_" in name and (lang == "en" or "q_proj" in name or "v_proj" in name):
            p.requires_grad = True
            n += p.numel()
    return n


# ------------------------------------------------------------------ the full model
def encode_targets(tok, texts, eos, device, max_len=64):
    ids = [tok(" " + t, add_special_tokens=False)["input_ids"][:max_len - 1] + [eos] for t in texts]
    L = max(len(i) for i in ids)
    t = torch.full((len(ids), L), eos, dtype=torch.long)
    m = torch.zeros((len(ids), L), dtype=torch.long)
    for r, i in enumerate(ids):
        t[r, :len(i)] = torch.tensor(i)
        m[r, :len(i)] = 1
    return t.to(device), m.to(device)


class SignLLM(nn.Module):
    def __init__(self, llm, tok, lang, n_tokens=16, enc_dim=256, enc_layers=3):
        super().__init__()
        self.llm, self.tok, self.lang = llm, tok, lang
        self.n_tokens, self.enc_dim, self.enc_layers = n_tokens, enc_dim, enc_layers
        emb = llm.get_input_embeddings()
        D = emb.weight.shape[1]
        std = float(emb.weight[:20000].detach().float().std())
        self.encoder = SignEncoder(d=enc_dim, layers=enc_layers)
        self.projector = LayerFusionProjector(enc_dim, enc_layers, D, n_tokens, init_scale=std)
        self.bos, self.eos = tok.bos_token_id, tok.eos_token_id
        self.prompt_ids = tok(PROMPT[lang], add_special_tokens=False)["input_ids"]

    def _prefix(self, feats):
        """[BOS][K visual tokens][prompt]  as embeddings."""
        dev = feats.device
        B = feats.size(0)
        vis = self.projector(self.encoder(feats))
        emb = self.llm.get_input_embeddings()
        bos = emb(torch.tensor([[self.bos]], device=dev)).expand(B, -1, -1)
        pr = emb(torch.tensor([self.prompt_ids], device=dev)).expand(B, -1, -1)
        return torch.cat([bos, vis.to(bos.dtype), pr], dim=1)

    def _full(self, feats, tgt_ids, tgt_mask):
        pre = self._prefix(feats)
        B, Lp, _ = pre.shape
        dev = pre.device
        x = torch.cat([pre, self.llm.get_input_embeddings()(tgt_ids)], dim=1)
        am = torch.cat([torch.ones(B, Lp, dtype=torch.long, device=dev), tgt_mask], dim=1)
        return x, am, Lp

    def forward(self, feats, tgt_ids, tgt_mask):
        """Training loss (only the target sentence + EOS is scored)."""
        x, am, Lp = self._full(feats, tgt_ids, tgt_mask)
        B = feats.size(0)
        labels = torch.cat([torch.full((B, Lp), -100, dtype=torch.long, device=x.device),
                            tgt_ids.masked_fill(tgt_mask == 0, -100)], dim=1)
        return self.llm(inputs_embeds=x, attention_mask=am, labels=labels).loss

    @torch.no_grad()
    def confidence(self, feats, tgt_ids, tgt_mask):
        """Geometric-mean probability of the given target tokens, per sequence."""
        x, am, Lp = self._full(feats, tgt_ids, tgt_mask)
        logits = self.llm(inputs_embeds=x, attention_mask=am).logits[:, Lp - 1:-1]
        lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        lp = (lp * tgt_mask).sum(1) / tgt_mask.sum(1).clamp(min=1)
        return lp.exp()

    @torch.no_grad()
    def generate(self, feats, max_new_tokens=60, num_beams=4):
        pre = self._prefix(feats)
        am = torch.ones(pre.shape[:2], dtype=torch.long, device=pre.device)
        out = self.llm.generate(inputs_embeds=pre, attention_mask=am, max_new_tokens=max_new_tokens,
                                num_beams=num_beams, do_sample=False,
                                eos_token_id=self.eos, pad_token_id=self.eos)
        texts = [t.strip() for t in self.tok.batch_decode(out, skip_special_tokens=True)]
        mask = torch.ones_like(out)
        for r in range(out.size(0)):
            pos = (out[r] == self.eos).nonzero()
            if len(pos):
                mask[r, pos[0, 0] + 1:] = 0
        return out, mask, texts


# ------------------------------------------------------------------ checkpoints (small: encoder, projector, LoRA)
def save_ckpt(path, sl, meta):
    lora = {n: p.detach().cpu() for n, p in sl.llm.named_parameters() if p.requires_grad and "lora_" in n}
    torch.save({"encoder": sl.encoder.state_dict(), "projector": sl.projector.state_dict(),
                "lora": lora, "meta": meta}, path)


def load_ckpt(path, sl):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sl.encoder.load_state_dict(ck["encoder"])
    sl.projector.load_state_dict(ck["projector"])
    params = dict(sl.llm.named_parameters())
    for n, t in ck["lora"].items():
        params[n].data.copy_(t.to(params[n].dtype))
    return ck["meta"]
