# SpecOACM-Det

Official implementation of **SpecOACM-Det**, a rotation-equivariant oriented
object detector built on C4 orbit-spectral decomposition and Mamba state-space
modeling.

This repository releases the core method modules. Training data, pretrained
checkpoints, data pipelines and the full hyper-parameter set are not included.

## Structure

```
SpecOACM-Det/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── configs/
│   ├── dota/base.py
│   ├── dior_r/base.py
│   └── hrsc2016/base.py
├── specoacm/
│   ├── convmamba.py           RMSNorm + SpecConvMamba
│   ├── direction_folding.py   C4 rotation, fold / unfold, spectral transform
│   ├── oacm.py                OACMBlock
│   ├── specoacm.py            SpecOACMBlock, SpecOACMBlockLite
│   ├── neck.py                OAFPN, SpecOAFPN
│   └── detector.py            Backbone, FPN, Detector
└── tools/
    ├── train.py
    └── test.py
```

`detector.py` is a reduced reference assembly used to show how the modules plug
together. It is not the exact network reported in the paper, so its parameter
count should not be compared with the paper tables.

## Method

| Module | Paper | Description |
|---|---|---|
| `convmamba.py` | §3.2 | Shared direction-aware ConvMamba operator |
| `direction_folding.py` | §3.3, §3.5 | `rot90` orbit, direction folding, C4 real-form spectral matrix |
| `oacm.py` | §3.3 | Four-direction rotation → shared Mamba → inverse rotation → mean |
| `specoacm.py` | §3.4, §3.6 | Orbit-spectral decomposition + invariant energy readout |
| `neck.py` | §3.7 | Direction-aware FPN injection |
| `detector.py` | — | Minimal backbone + neck + head assembly |

### Direction folding

The four canonical rotations are batched as `(4B, C, H, W)` and evaluated by a
single Mamba call:

```python
orbit = unfold(self.mamba(fold(x, self.angles)), self.angles)
```

This is mathematically equivalent to looping over the four directions, with
fewer kernel launches and no accuracy change.

### Spectral readout

`c4_spectral` applies the fixed real-form C4 matrix to the orbit, producing
`Z0, Z1c, Z1s, Z2`. The rotation-invariant magnitudes `E1 = sqrt(Z1c² + Z1s² +
eps)` and `E2 = |Z2|` are fused by a two-layer pointwise branch whose second
convolution is zero-initialized, so the block starts as a strict identity:

```python
return x + self.alpha0 * z0 + energy
```

## Usage

```python
import torch
from specoacm import SpecOACMBlock, SpecOACMBlockLite

block = SpecOACMBlock(dim=256, num_dirs=4, d_state=16, expand=2)
y = block(torch.randn(2, 256, 50, 50))

lite = SpecOACMBlockLite(dim=256, num_dirs=4, d_state=16)
y = lite(torch.randn(2, 256, 50, 50))
```

Full detector:

```python
from specoacm import Detector

model = Detector(
    num_classes=20,
    base_channels=64,
    base_depth=3,
    inject_indices=(2, 3),
    neck_levels=(False, True, True),
    block='SpecOACMBlock',
    block_cfg=dict(num_dirs=4, d_state=16, d_conv=4, expand=2),
)
```

## Installation

```
pip install -r requirements.txt
```

or

```
conda env create -f environment.yml
```

`mamba-ssm` requires a CUDA toolkit matching the installed PyTorch build.

## Citation

```
@article{specoacmdet,
  title   = {SpecOACM-Det: Rotation-Equivariant Oriented Detection via
             Orbit-Spectral Mamba},
  author  = {Anonymous},
  year    = {2026}
}
```
