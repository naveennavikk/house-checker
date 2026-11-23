"""Core logic: decide whether an image shows a house with a visible door or gate.
 
How it works
------------
1. A pretrained CLIP image model (DataComp CLIP ViT-B/32) turns the photo into a
   512-number "embedding" that describes what is in the picture.
2. A small classifier (model/probe.json), trained on ~4,000 labelled survey photos
   (accepted property images vs "property image not proper", long-distance,
   washroom and plate close-ups), turns that embedding into a probability.
3. That is blended 50/50 with a vote of the 15 most similar labelled photos in
   model/reference_bank.npz. Add your own corrections with add_examples.py.
4. Score >= threshold (stored in probe.json) -> ACCEPT, otherwise -> REJECT.
 
A house entrance doorway seen from outside is ACCEPTED. Interiors, door-detail
close-ups (handles, locks), houses with no visible door, and
unrelated photos are all REJECTED.
"""
 
from __future__ import annotations
 
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
import torch
from PIL import Image, ImageOps
 
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
PROBE_PATH = MODEL_DIR / "probe.json"
ENCODER_PATH = MODEL_DIR / "clip_b32_datacomp.ts"
BANK_PATH = MODEL_DIR / "reference_bank.npz"
ENCODER_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/clip_b32_datacomp.ts"
ENCODER_SHA256 = "ffd6ea9ca4cd53831eae55a60168eb5cf97ddbb8d53bed17d8bf063d42c1d8b1"
 
 
@dataclass
class Result:
    accepted: bool
    probability: float  # 0..1, how sure the model is that it should be ACCEPTED
 
    @property
    def label(self) -> str:
        return "ACCEPT" if self.accepted else "REJECT"
 
 
def _download_encoder() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ENCODER_PATH.with_suffix(".part")
    print(f"Downloading the image model (about 290 MB, one time only) from\n  {ENCODER_URL}", file=sys.stderr)
 
    def hook(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  {pct}%", end="", file=sys.stderr, flush=True)
 
    urllib.request.urlretrieve(ENCODER_URL, tmp, reporthook=hook)
    print(file=sys.stderr)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != ENCODER_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded model file is corrupted (checksum mismatch). Please try again.")
    tmp.replace(ENCODER_PATH)
 
 
class HouseDoorChecker:
    def __init__(self, threshold: float | None = None):
        if not ENCODER_PATH.exists():
            _download_encoder()
        self.encoder = torch.jit.load(str(ENCODER_PATH), map_location="cpu").float().eval()
        probe = json.loads(PROBE_PATH.read_text())
        self.w = np.asarray(probe["weights"], dtype=np.float32)
        self.b = float(probe["bias"])
        self.version = int(probe.get("version", 1))
        self.threshold = float(threshold if threshold is not None else probe["threshold"])
        # v2: zero-shot text prompts (pre-computed) blended with the trained classifier.
        # This helps with photo styles that were rare in the training photos,
        # e.g. Indian house entrances with a garland, open doors, survey/geotag text.
        self.blend = float(probe.get("blend_zero_shot", 0.0))
        self.prompt_emb = np.asarray(probe.get("prompt_embeddings", []), dtype=np.float32)
        self.prompt_pos = np.asarray(probe.get("prompt_is_positive", []), dtype=bool)
        # v5: nearest-neighbour vote over a bank of labelled survey photos
        # (model/reference_bank.npz). Photos added with add_examples.py go in here too.
        self.knn_weight = float(probe.get("knn_weight", 0.0))
        self.knn_k = int(probe.get("knn_k", 15))
        self.knn_temp = float(probe.get("knn_temp", 0.05))
        self.bank_emb = self.bank_label = None
        if self.knn_weight > 0 and BANK_PATH.exists():
            bank = np.load(BANK_PATH)
            self.bank_emb = bank["emb"].astype(np.float32)
            self.bank_label = bank["label"].astype(np.float32)

    def embed(self, image: str | Path | Image.Image) -> np.ndarray:
        """Unit-length CLIP embedding of an image."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        with torch.inference_mode():
            emb = self.encoder.encode_image(self._preprocess(image))[0].numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

    def _knn_score(self, emb: np.ndarray) -> float:
        """Similarity-weighted vote of the most similar labelled photos (class-balanced)."""
        sim = self.bank_emb @ emb
        idx = np.argsort(-sim)[: self.knn_k]
        s = sim[idx]
        labels = self.bank_label[idx]
        prior = float(self.bank_label.mean())
        wt = np.exp((s - s[0]) / self.knn_temp) * np.where(labels == 1, 0.5 / prior, 0.5 / (1 - prior))
        return float((wt * labels).sum() / wt.sum())
 
    @staticmethod
    def _preprocess(img: Image.Image) -> torch.Tensor:
        img = ImageOps.exif_transpose(img).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, 0..1
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
 
    def check(self, image: str | Path | Image.Image) -> Result:
        emb = self.embed(image)
        prob = float(1.0 / (1.0 + np.exp(-(emb @ self.w + self.b))))
        if self.bank_emb is not None:
            prob = (1.0 - self.knn_weight) * prob + self.knn_weight * self._knn_score(emb)
        if self.blend > 0 and len(self.prompt_emb):
            logits = 100.0 * (self.prompt_emb @ emb)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            zero_shot = float(p[self.prompt_pos].sum())
            prob = (1.0 - self.blend) * prob + self.blend * zero_shot
        return Result(accepted=prob >= self.threshold, probability=prob)
 