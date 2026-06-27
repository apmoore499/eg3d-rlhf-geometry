# Third-party notices

This project is released under the NVIDIA Source Code License for EG3D (see
`LICENSE.txt`), reflecting its origin as a fork of [EG3D](https://github.com/NVlabs/eg3d).
It also incorporates the components below, each under its own terms. The license
text for vendored components is kept alongside the code in the paths listed; the
templates and references that are not shipped with a separate license file are
reproduced here.

## Scaffolding

The reward-model training framework (`reward_model_training/reward_model_framework/`)
was built on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)
by ashleve, used under the MIT License:

```
MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Vendored components

Each retains its own license file at the path shown.

| Component | License | Copyright | Path |
|-----------|---------|-----------|------|
| [EG3D](https://github.com/NVlabs/eg3d) | NVIDIA Source Code License (non-commercial) | NVIDIA Corporation | `LICENSE.txt`, `eg3d/` |
| [PointNet](https://github.com/charlesq34/pointnet) | MIT | 2017 Geometric Computation Group, Stanford | `external/pointnet/LICENSE` |
| [PointContrast](https://github.com/facebookresearch/PointContrast) | MIT | Facebook, Inc. | `external/PointContrast/LICENSE` |
| [AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss) | MIT | 2019 Elliott Zheng | `external/AdaptiveWingLoss/LICENSE` |
| [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) | MIT | 2020 NVIDIA; 2018–2020 Chris Choy | `eg3d/external_modules/MinkowskiEngine/LICENSE` |
| sam | MIT | 2021 David Samuel | `eg3d/external_modules/sam/LICENSE` |
| [CurveNet](https://github.com/tiangexiang/CurveNet) | MIT | 2021 Tiange Xiang | (point-cloud backbone, referenced) |
| [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) | MIT | 2019 Adrian Wolny | 3D U-Net backbones (`pytorch3dunet` dependency) |

## Datasets

The [Flickr-Faces-HQ (FFHQ)](https://github.com/NVlabs/ffhq-dataset) dataset is
**not** redistributed here. Its preprocessing code is included; FFHQ itself is
made available by NVIDIA under a non-commercial research license (see
`dataset_preprocessing/ffhq/LICENSE.txt`).
