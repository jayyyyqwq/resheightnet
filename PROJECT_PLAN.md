# ResHeightNet — Project Plan

**A PyTorch reimplementation of "Height Estimation from Single Aerial Images Using a Deep Convolutional Encoder-Decoder Network" (Amirkolaee & Arefi, ISPRS Journal of Photogrammetry and Remote Sensing, 2019), built as a standalone, from-scratch, transfer-learning-based baseline.**

This document is the single source of truth for building this project. It explains what the project is, why each piece exists, exactly how to implement it, how to run it, and how to eventually host it. Follow it top to bottom and you will have a working repo.

---

## Table of Contents

1. [Overview & Goals](#1-overview--goals)
2. [Relationship to the Original Paper](#2-relationship-to-the-original-paper)
3. [Relationship to DepthWizard](#3-relationship-to-depthwizard)
4. [Architecture](#4-architecture)
5. [Repository Structure](#5-repository-structure)
6. [Environment Setup](#6-environment-setup)
7. [Dataset & Preprocessing](#7-dataset--preprocessing)
8. [Model Implementation](#8-model-implementation)
9. [Training](#9-training)
10. [Evaluation](#10-evaluation)
11. [Inference / Demo Script](#11-inference--demo-script)
12. [Implementation Roadmap (Milestones)](#12-implementation-roadmap-milestones)
13. [README Template](#13-readme-template)
14. [Hosting & Deployment (Low Priority)](#14-hosting--deployment-low-priority)
15. [Attribution & Honesty Notes](#15-attribution--honesty-notes)
16. [Quick Reference Commands](#16-quick-reference-commands)

---

## 1. Overview & Goals

**What this is:** a standalone GitHub repository that reimplements a real, cited, peer-reviewed remote-sensing paper end to end: single RGB aerial image in, predicted height/nDSM map out. No official code exists for this paper anywhere, so this is a genuine open-source contribution, not a clone of someone else's repo.

**What "done" looks like** (the actual finish line, kept realistic):

- A model that runs on a single GPU (or CPU, slowly) and produces visibly reasonable height maps on held-out test tiles.
- A documented training run with loss curves and final metrics (MAE, RMSE, correlation).
- A comparison table against DepthWizard's own numbers, using the same evaluation split and metrics.
- A README a stranger could read and understand, and a live inference demo (Milestone 6, optional).

**What this is explicitly not:** an attempt to match the original paper's published accuracy on the original ISPRS benchmark. That paper trained a custom encoder from scratch on Vaihingen/Potsdam over many epochs. This project uses transfer learning and an already-available dataset (GAMUS) to make the build tractable in limited time. That's a deliberate, disclosed simplification, not a hidden shortcut. See [Section 15](#15-attribution--honesty-notes).

---

## 2. Relationship to the Original Paper

The paper proposes a CNN with:

- An **encoder** built on deep residual learning (ResNet-style blocks) to extract local and global features from a single RGB image.
- A **decoder** that upsamples those features back to full resolution, using skip connections at each scale to sharpen object boundaries (buildings, edges) in the predicted height map.
- Training on ISPRS Vaihingen/Potsdam, using LiDAR-derived DSM as ground truth.

**What we keep faithful:**
- The encoder-decoder shape with skip connections.
- Regression to a continuous height value per pixel (not classification/discretization).
- Standard reconstruction-error evaluation (MAE, RMSE).

**What we deliberately change (and disclose):**
- Encoder: pretrained ImageNet ResNet34 (via `torchvision`) instead of a custom residual encoder trained from scratch. This is the single biggest reason this project is achievable on a deadline: you are fine-tuning, not learning feature extraction from zero.
- Dataset: GAMUS (already integrated into your DepthWizard pipeline) instead of acquiring and preprocessing ISPRS Vaihingen/Potsdam from scratch.

---

## 3. Relationship to DepthWizard

This repo is intentionally **separate** from the main DepthWizard repository, but it borrows two things directly from it to save time:

1. **Data pipeline reuse:** point this project's data loader at the same GAMUS deterministic splits DepthWizard already uses (`data/gamus/splits/`). Copy or symlink them in; do not re-derive new splits.
2. **Evaluation parity:** use the exact same metrics DepthWizard reports (Mean Absolute Height Error, RMSE, Ground-Truth Correlation) so the two systems are directly comparable in a results table. This is what makes the repo useful as a credibility artifact, not just an isolated side project — it produces a real, apples-to-apples baseline number you can cite next to DepthWizard's own results.

No model code, weights, or DepthWizard-specific architecture (RDAH, CBAM, frozen Depth Anything V2) is shared. This project is a clean-room baseline reimplementation, which is exactly what makes the comparison meaningful.

---

## 4. Architecture

```
Input RGB image (3 x H x W)
        │
        ▼
┌───────────────────────────┐
│  ResNet34 Encoder          │   pretrained on ImageNet
│  (stem frozen, layer3/4    │
│   fine-tuned)               │
│                             │
│  stage1 ──┐                 │  64  channels,  H/4
│  stage2 ──┼─┐               │  128 channels,  H/8
│  stage3 ──┼─┼─┐             │  256 channels,  H/16
│  stage4 ──┼─┼─┼─┐           │  512 channels,  H/32
└───────────┼─┼─┼─┼───────────┘
            │ │ │ │  (skip connections, one per stage)
            ▼ ▼ ▼ ▼
┌───────────────────────────┐
│  Decoder                   │
│  4x [Upsample → Concat     │
│       skip → Conv → BN →   │
│       ReLU]                 │
└─────────────┬───────────────┘
              ▼
      1x1 Conv → single-channel
       height map (H x W)
```

**Loss function (start simple, add complexity only if needed):**
- Phase 1: plain **L1 loss** (Mean Absolute Error) between predicted and ground-truth height.
- Phase 2 (optional refinement): add an edge-aware smoothness term so predicted height doesn't blur across building boundaries — this mirrors what the paper's skip connections are trying to preserve. Only attempt this after Phase 1 works end to end.

---

## 5. Repository Structure

```
resheightnet/
├── README.md
├── PROJECT_PLAN.md              <- this document
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   └── splits/                  <- copied/symlinked from DepthWizard's GAMUS splits
├── src/
│   ├── dataset.py                <- GAMUS loader + preprocessing
│   ├── model.py                  <- ResNet34 encoder + decoder
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── utils.py
├── notebooks/
│   └── 01_quick_demo.ipynb       <- for visual sanity checks during dev
├── web_demo/                     <- Milestone 6, lowest priority
│   └── app.py
├── results/
│   ├── checkpoints/
│   └── sample_outputs/
└── docs/
    └── comparison_table.md       <- ResHeightNet vs DepthWizard results
```

---

## 6. Environment Setup

`requirements.txt`:
```
torch>=2.1
torchvision>=0.16
numpy
pillow
rasterio
matplotlib
pyyaml
tqdm
scikit-learn
```

Setup (Linux/macOS):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Setup (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you don't have a local GPU, Google Colab's free T4 GPU is enough for this model at moderate patch sizes (e.g. 256x256 or 384x384).

---

## 7. Dataset & Preprocessing

- **Source:** GAMUS RGB + AGL/nDSM pairs, using the deterministic splits already defined for DepthWizard. Do not create new splits — reuse the existing train/val/test lists so results are directly comparable.
- **Patch size:** start with 256x256 random crops for training (faster iteration, less memory), evaluate on full tiles or larger crops (e.g. 512x512).
- **Normalization:** RGB normalized with ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`) since the encoder is ImageNet-pretrained.
- **Height target:** keep in meters directly for Phase 1. If training is unstable, a log1p transform on the target (and inverse on prediction) often helps regression stability for skewed height distributions — treat this as an optional Phase 2 tweak, not a Phase 1 requirement.
- **Augmentation (keep minimal at first):** random horizontal/vertical flip, random 90-degree rotation. Skip color jitter initially — it can hurt height regression by disturbing brightness cues the model uses.

`src/dataset.py` skeleton:
```python
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class GamusHeightDataset(Dataset):
    def __init__(self, split_file, patch_size=256, train=True):
        with open(split_file) as f:
            self.samples = [line.strip().split(",") for line in f if line.strip()]
        self.patch_size = patch_size
        self.train = train
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, height_path = self.samples[idx]

        with rasterio.open(rgb_path) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0  # C,H,W

        with rasterio.open(height_path) as src:
            height = src.read(1).astype(np.float32)  # H,W

        rgb_t = torch.from_numpy(rgb)
        height_t = torch.from_numpy(height).unsqueeze(0)

        rgb_t, height_t = self._crop(rgb_t, height_t)

        if self.train:
            rgb_t, height_t = self._augment(rgb_t, height_t)

        rgb_t = self.normalize(rgb_t)
        return rgb_t, height_t

    def _crop(self, rgb, height):
        _, h, w = rgb.shape
        p = self.patch_size
        if h < p or w < p:
            pad_h, pad_w = max(0, p - h), max(0, p - w)
            rgb = torch.nn.functional.pad(rgb, (0, pad_w, 0, pad_h))
            height = torch.nn.functional.pad(height, (0, pad_w, 0, pad_h))
            _, h, w = rgb.shape
        top = torch.randint(0, h - p + 1, (1,)).item() if self.train else (h - p) // 2
        left = torch.randint(0, w - p + 1, (1,)).item() if self.train else (w - p) // 2
        return rgb[:, top:top+p, left:left+p], height[:, top:top+p, left:left+p]

    def _augment(self, rgb, height):
        if torch.rand(1).item() > 0.5:
            rgb, height = torch.flip(rgb, dims=[2]), torch.flip(height, dims=[2])
        if torch.rand(1).item() > 0.5:
            rgb, height = torch.flip(rgb, dims=[1]), torch.flip(height, dims=[1])
        return rgb, height
```

This is a working starting point, not a finished, edge-case-proof loader. Expect to adjust paths/format handling to match how your GAMUS split files are actually structured.

---

## 8. Model Implementation

`src/model.py`:
```python
import torch
import torch.nn as nn
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResHeightNet(nn.Module):
    """
    ResNet34 encoder + skip-connected decoder for single-image height regression.
    Adapted from Amirkolaee & Arefi (2019), ISPRS J. Photogramm. Remote Sens.
    """
    def __init__(self, pretrained=True, freeze_stem=True):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   # H/2,  64ch
        self.pool = resnet.maxpool                                         # H/4
        self.layer1 = resnet.layer1   # H/4,  64ch
        self.layer2 = resnet.layer2   # H/8,  128ch
        self.layer3 = resnet.layer3   # H/16, 256ch
        self.layer4 = resnet.layer4   # H/32, 512ch

        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad = False

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)

        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        s0 = self.stem(x)          # H/2
        p0 = self.pool(s0)         # H/4
        s1 = self.layer1(p0)       # H/4
        s2 = self.layer2(s1)       # H/8
        s3 = self.layer3(s2)       # H/16
        s4 = self.layer4(s3)       # H/32

        d4 = self.dec4(s4, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)

        out = self.final_up(d1)
        out = nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(out)
```

This is a genuine, runnable architecture, not pseudocode. It will need debugging once wired into your actual data shapes, but the design is sound and follows the paper's structure closely.

---

## 9. Training

`src/train.py` (skeleton, fill in config loading and checkpointing to taste):
```python
import torch
from torch.utils.data import DataLoader
from model import ResHeightNet
from dataset import GamusHeightDataset

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = GamusHeightDataset("data/splits/train.txt", patch_size=256, train=True)
    val_ds = GamusHeightDataset("data/splits/val.txt", patch_size=256, train=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = ResHeightNet(pretrained=True, freeze_stem=True).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = torch.nn.L1Loss()

    best_val = float("inf")
    for epoch in range(30):
        model.train()
        train_loss = 0.0
        for rgb, height in train_loader:
            rgb, height = rgb.to(device), height.to(device)
            optimizer.zero_grad()
            pred = model(rgb)
            loss = criterion(pred, height)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb, height in val_loader:
                rgb, height = rgb.to(device), height.to(device)
                pred = model(rgb)
                val_loss += criterion(pred, height).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"epoch {epoch+1:02d}  train_L1={train_loss:.3f}  val_L1={val_loss:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "results/checkpoints/best.pth")

if __name__ == "__main__":
    train()
```

**Practical training notes:**
- Batch size 8 at 256x256 fits comfortably on a T4 (16GB). Drop to 4 if you hit out-of-memory errors.
- Start with a tiny subset (a few hundred patches) and confirm loss goes down before committing to a full run. This is Milestone 2 below, do not skip it.
- 20-30 epochs on a modest subset is a realistic target for a hackathon-scale demonstration, not hundreds of epochs.

---

## 10. Evaluation

`src/evaluate.py` should report the same three metrics DepthWizard reports, so results are directly comparable:

- **Mean Absolute Height Error (MAE)**
- **RMSE**
- **Ground-Truth Correlation** (Pearson correlation between predicted and true height values, pixel-wise or per-tile-mean, matching however DepthWizard computed its 68% figure)

```python
import torch
import numpy as np
from scipy.stats import pearsonr

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for rgb, height in loader:
            rgb = rgb.to(device)
            pred = model(rgb).cpu().numpy().flatten()
            gt = height.numpy().flatten()
            all_preds.append(pred)
            all_gts.append(gt)

    preds = np.concatenate(all_preds)
    gts = np.concatenate(all_gts)

    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts) ** 2))
    corr, _ = pearsonr(preds, gts)

    return {"MAE": mae, "RMSE": rmse, "Correlation": corr}
```

Write the result into `docs/comparison_table.md`:

```markdown
| Model        | MAE (m) | RMSE (m) | Correlation |
|--------------|---------|----------|-------------|
| DepthWizard  | 3.31    | 6.28     | 0.68        |
| ResHeightNet | TBD     | TBD      | TBD         |
```

Fill in ResHeightNet's row once you have real numbers. Never estimate or guess a placeholder number into this table.

---

## 11. Inference / Demo Script

`src/infer.py`:
```python
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import ResHeightNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="results/checkpoints/best.pth")
    parser.add_argument("--out", default="results/sample_outputs/out.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResHeightNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    img = Image.open(args.image).convert("RGB")
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensor = transforms.ToTensor()(img)
    tensor = normalize(tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()

    plt.imsave(args.out, pred, cmap="viridis")
    print(f"Saved height map to {args.out}")

if __name__ == "__main__":
    main()
```

Run with:
```bash
python src/infer.py --image path/to/test_tile.tif --checkpoint results/checkpoints/best.pth
```

---

## 12. Implementation Roadmap (Milestones)

Given you're still building up DL fundamentals, work through these in strict order. Each one is a real, checkable stopping point, don't skip ahead until the current one visibly works.

| Milestone | Goal | Signal it's done |
|-----------|------|-------------------|
| **M0** | Environment + data loader running | You can load one batch and `plt.imshow()` both the RGB and the height map without errors |
| **M1** | Model forward pass | A random tensor of the right shape passes through `ResHeightNet` and produces an output of matching H/W with no shape errors |
| **M2** | Overfit a tiny subset | Train on 5-10 images repeatedly; loss should drop close to zero. This proves the model and loss are wired correctly before you trust a real run |
| **M3** | Full training run | A real training run on your GAMUS train split subset completes and saves a checkpoint |
| **M4** | Evaluation | `evaluate.py` produces real MAE/RMSE/correlation numbers on the val/test split |
| **M5** | Inference + visuals | `infer.py` produces a visually sensible height map on a held-out test image; drop 3-4 example images into the README |
| **M6 (stretch)** | Web demo + hosting | Gradio app wrapping `infer.py`, deployed to Hugging Face Spaces, link added to resume/README |

Do not attempt M6 before M5 is genuinely done. A working local demo beats a broken public one.

---

## 13. README Template

Use this as the actual `README.md` once the project works:

```markdown
# ResHeightNet

A PyTorch reimplementation of *"Height Estimation from Single Aerial Images Using a Deep
Convolutional Encoder-Decoder Network"* (Amirkolaee & Arefi, 2019, ISPRS Journal of
Photogrammetry and Remote Sensing), built as an independent baseline using transfer learning.

No official code was released for this paper — this is an open, from-scratch implementation.

## What it does
Given a single RGB aerial/satellite image, predicts a per-pixel height map (nDSM).

## Architecture
ResNet34 encoder (ImageNet-pretrained) + skip-connected upsampling decoder.
See [PROJECT_PLAN.md](PROJECT_PLAN.md) for full architecture details and rationale.

## Results
| Model        | MAE (m) | RMSE (m) | Correlation |
|--------------|---------|----------|-------------|
| ResHeightNet | ...     | ...      | ...         |

Trained and evaluated on the GAMUS dataset (same split used by [DepthWizard](link)).

## Usage
\`\`\`bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
python src/infer.py --image sample.tif
\`\`\`

## Honest scope note
This reimplementation uses a pretrained ResNet34 encoder and the GAMUS dataset rather than
training a custom encoder from scratch on ISPRS Vaihingen/Potsdam as the original paper did.
Architecture and training objective follow the paper closely; exact published numbers are not
expected to be reproduced.

## Citation
Amirkolaee, H. A., & Arefi, H. (2019). Height estimation from single aerial images using a deep
convolutional encoder-decoder network. *ISPRS Journal of Photogrammetry and Remote Sensing*, 149, 50-66.

## Live demo
[link once hosted]
```

---

## 14. Hosting & Deployment (Low Priority)

Only attempt this after M5 is solid.

**Recommended path: Hugging Face Spaces + Gradio**
- Free, no server management, generates a real public URL you can put on a resume.
- Wrap `infer.py`'s logic in a small Gradio interface: image upload in, height-map image out.
- Steps: create a Space (Gradio SDK), push `app.py` + `requirements.txt` + your checkpoint (or a small demo checkpoint if the full one is too large), done.

```python
# web_demo/app.py (skeleton)
import gradio as gr
import torch
from src.model import ResHeightNet
from torchvision import transforms
import numpy as np

device = "cpu"
model = ResHeightNet(pretrained=False)
model.load_state_dict(torch.load("results/checkpoints/best.pth", map_location=device))
model.eval()

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def predict(image):
    tensor = normalize(transforms.ToTensor()(image)).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor).squeeze().numpy()
    return (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)

demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs=gr.Image(type="numpy"))
demo.launch()
```

Once live, add the link to the README's "Live demo" section and, if you want, your resume/portfolio.

---

## 15. Attribution & Honesty Notes

- Always cite the original paper prominently. This is a reimplementation, not your own novel research contribution, and the README should say so plainly.
- Be explicit about every deviation from the original method (pretrained encoder, different dataset). This is a strength, not a weakness, when disclosed. It shows you understand the paper well enough to know what you're changing and why.
- Never fill the results table with estimated or placeholder numbers. If a run isn't done yet, leave it as "TBD" or "in progress," not a guess.
- If asked in a judging context "did you exactly reproduce the paper," the honest answer is: "no, we adapted it with transfer learning and a different dataset to make it feasible on our timeline, and we're transparent about that in the README." That answer is more credible than an inflated claim.

---

## 16. Quick Reference Commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# sanity check data loading (M0)
python -c "from src.dataset import GamusHeightDataset; d = GamusHeightDataset('data/splits/train.txt'); print(d[0][0].shape, d[0][1].shape)"

# sanity check model forward pass (M1)
python -c "import torch; from src.model import ResHeightNet; m = ResHeightNet(); print(m(torch.randn(1,3,256,256)).shape)"

# train
python src/train.py

# evaluate
python src/evaluate.py

# run inference on one image
python src/infer.py --image path/to/test_tile.tif --out results/sample_outputs/demo.png
```
