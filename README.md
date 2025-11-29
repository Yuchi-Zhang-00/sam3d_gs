<p align="center">
  <a href="README_zh.md">
    <img src="https://img.shields.io/badge/语言-中文-red?style=for-the-badge">
  </a>
</p>

# **Unified Multi-Stage 2D→3D Perception Pipeline**

## *vLLM × SAM3 × SAM-3D-Objects Integration*

------

## **Abstract**

This repository presents a unified and modular pipeline that couples large-scale vision–language reasoning, high-fidelity 2D segmentation, and multi-object 3D Gaussian splatting. It integrates three independent systems—**vLLM** (for Qwen3-VL inference), **SAM3** (for multi-object 2D segmentation), and **SAM-3D-Objects** (for 3D reconstruction from RGB + masks)—into a complete, end-to-end workflow. To ensure reproducibility, each module runs inside its own Conda environment. The pipeline supports both staged execution and a fully automated one-click execution, with built-in HuggingFace authentication, checkpoint management, and environment initialization.

------

# **1. Repository Setup**

```
git clone --recursive https://github.com/xyys2003/sam3d_gs.git
cd sam3d_gs
```

If cloned without submodules:

```
git submodule update --init --recursive
```

------

# **2. Conda Environments**

| Environment     | Purpose                                  | Path              |
| --------------- | ---------------------------------------- | ----------------- |
| `vllm`          | Serve Qwen3-VL-8B-Thinking via vLLM      | —                 |
| `sam3`          | Multi-object segmentation (SAM3)         | `sam3/`           |
| `sam3d-objects` | RGB + masks → 3D Gaussian reconstruction | `sam-3d-objects/` |

------

# **3. vLLM Environment (Qwen3-VL Server)**

```
conda create -n vllm python=3.10 -y
conda activate vllm
```

Install PyTorch (CUDA 12.x):

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

Install vLLM:

```
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124
pip install transformers tiktoken sentencepiece xformers flashinfer-python
pip install huggingface_hub
```

------

# **4. SAM3 Environment**

Reference implementation:
 🔗 https://github.com/facebookresearch/sam3
 🔗 https://huggingface.co/facebook/sam3

```
cd sam3
conda create -n sam3 python=3.10 -y
conda activate sam3
```

Install SAM3:

```
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

Optional:

```
pip install -e ".[notebooks]"
pip install -e ".[train,dev]"
```

------

# **5. SAM-3D-Objects Environment**

Reference implementation:
 🔗 https://github.com/facebookresearch/sam3d
 🔗 https://huggingface.co/facebook/sam-3d-objects

```
conda create -n sam_3d_body python=3.10 -y
conda activate sam_3d_body
```

Install dependencies (excerpt):

```
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas hydra-core ...
```

Install Detectron2:

```
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps
```

Optional: MoGe

```
pip install git+https://github.com/microsoft/MoGe.git
```

------

# **6. Required HuggingFace Access**

The pipeline requires access to the following models:

- **SAM3**
   🔗 https://huggingface.co/facebook/sam3
- **SAM-3D-Objects**
   🔗 https://huggingface.co/facebook/sam-3d-objects

Log in after requesting access:

```
hf auth login
```

------

# **7. Running the Pipeline**

Ensure the Conda activation path is correct:

```
CONDA_SH="/your_path/miniconda3/etc/profile.d/conda.sh"
```

------

## **Stage 1 — Qwen3-VL + SAM3 (2D Mask Generation)**

```
bash run_agent_with_vllm.sh
```

Outputs:

```
outputs/master_with_vllm/masks/
```

------

## **Stage 2 — SAM-3D-Objects Reconstruction**

```
bash run_sam3d_from_masks.sh
```

Outputs:

```
sam-3d-objects/outputs/torch_save_pt/
sam-3d-objects/gaussians/multi/
```

------

## **Optional: One-Click Execution**

```
bash run_pipeline.sh
```

------

# **8. Q&A**

## **Q1: Download error “Consistency check failed: file should be XXXX but has size YYYY”?**

Cause: corrupted model shards in the HuggingFace cache due to unstable network.

Fix:

```
rm -rf sam-3d-objects/checkpoints/hf
rm -rf ~/.cache/huggingface/hub   # optional
bash run_sam3d_from_masks.sh
```

Force fresh download:

```
force_download=True
```

## **Note on Coordinate System (PLY Output Orientation)**

The 3D Gaussian `.ply` files exported by **SAM-3D-Objects** are expressed in the **camera coordinate system**, where:

- **+Z axis** points **forward** from the camera
- **+X axis** points right
- **+Y axis** points downward (typical computer vision convention)

This means the reconstructed objects are aligned using **camera-forward Z-axis** rather than a world coordinate frame.

If you want to visualize or place the objects in a global **world coordinate system**, you must apply a **camera-to-world transformation**:
$$
\mathbf{X}_{world} = \mathbf{R}_{c2w}\ \mathbf{X}_{camera} \ + \ \mathbf{t}_{c2w}
$$
Where:

- $\mathbf{R}_{c2w}$ is the rotation matrix from camera to world
- $\mathbf{t}_{c2w}$ is the translation vector
- $\mathbf{X}_{camera}$ is the Gaussian center in camera coordinates
- $\mathbf{X}_{world}$ is the desired world coordinate position

After applying this transformation, the `.ply` will correctly align with your global scene, robotics simulator, or NeRF / COLMAP world frame.
------

# **Citation**

### SAM3

```
@article{kirillov2024sam3,
  title={SAM 3: Segment Anything in Images and Videos},
  author={Kirillov, Alexander and Ravi, Nikhila and Mao, Weiyao and others},
  year={2024},
  url={https://github.com/facebookresearch/sam3}
}
```

### SAM-3D-Objects

```
@article{wu2024sam3dobjects,
  title={SAM-3D-Objects: Segment Anything in 3D Using 2D Masks},
  author={Wu, Yu and Mao, Weiyao and Kirillov, Alexander and others},
  year={2024},
  url={https://github.com/facebookresearch/sam3d}
}
```

------

# **Acknowledgements**

This project is built upon and integrates:

- **SAM3**
   GitHub: https://github.com/facebookresearch/sam3
   HuggingFace: https://huggingface.co/facebook/sam3
- **SAM-3D-Objects**
   GitHub: https://github.com/facebookresearch/sam3d
   HuggingFace: https://huggingface.co/facebook/sam-3d-objects

We sincerely thank the authors for making their research and implementations publicly available.