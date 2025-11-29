<p align="center">
  <a href="README.md">
    <img src="https://img.shields.io/badge/Language-English-blue?style=for-the-badge">
  </a>
</p>

# **统一的多阶段 2D→3D 感知流水线**

## *vLLM × SAM3 × SAM-3D-Objects 集成*

------

## **摘要**

本仓库构建了一个完整的 2D → 3D 感知流水线，将 **大模型视觉理解、2D 多物体分割、3D Gaussian Splatting 重建** 三者进行统一整合。流水线由：

- **vLLM**：提供 Qwen3-VL-8B-Thinking 视觉语言大模型推理
- **SAM3**：执行高质量多物体 2D 分割
- **SAM-3D-Objects**：将 RGB + mask 提升为 3D 高斯点（Gaussian Splat）

为确保可复现性，每个模块均独立运行在各自的 Conda 环境中。系统支持 **分阶段执行**（先 2D 分割、再 3D 重建），也支持 **一键式全流程运行**。

------

# **1. 仓库克隆**

```
git clone --recursive https://github.com/xyys2003/sam3d_gs.git
cd sam3d_gs
```

如果你忘记使用 `--recursive` 克隆，可运行：

```
git submodule update --init --recursive
```

------

# **2. Conda 环境说明**

本项目使用三个互相隔离的 Conda 环境，以避免依赖冲突。

| 环境名称        | 功能用途                           | 路径              |
| --------------- | ---------------------------------- | ----------------- |
| `vllm`          | 运行 Qwen3-VL-8B-Thinking 推理服务 | —                 |
| `sam3`          | 运行 SAM3 完成 2D 多物体分割       | `sam3/`           |
| `sam3d-objects` | 从 RGB + Mask 生成 3D Gaussian     | `sam-3d-objects/` |

------

# **3. vLLM 环境（Qwen3-VL 服务器）**

### **3.1 创建环境**

```
conda create -n vllm python=3.10 -y
conda activate vllm
```

### **3.2 安装 PyTorch（CUDA 12.x）**

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

### **3.3 安装 vLLM 与相关依赖**

```
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124
pip install transformers tiktoken sentencepiece xformers flashinfer-python
pip install huggingface_hub
```

此配置已验证可稳定运行 **Qwen3-VL-8B-Thinking**。

------

# **4. SAM3 环境**

官方实现：
 🔗 https://github.com/facebookresearch/sam3
 🔗 https://huggingface.co/facebook/sam3

### **4.1 创建环境**

```
cd sam3
conda create -n sam3 python=3.10 -y
conda activate sam3
```

### **4.2 安装 PyTorch（CUDA 12.x）**

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

### **4.3 克隆并安装 SAM3**

```
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### **4.4 可选依赖（用于 Notebook 或训练）**

```
pip install -e ".[notebooks]"
pip install -e ".[train,dev]"
```

------

# **5. SAM-3D-Objects 环境**

官方实现：
 🔗 https://github.com/facebookresearch/sam3d
 🔗 https://huggingface.co/facebook/sam-3d-objects

### **5.1 创建环境**

```
conda create -n sam_3d_body python=3.10 -y
conda activate sam_3d_body
```

### **5.2 安装 PyTorch（CUDA 12.x）**

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

### **5.3 安装其他 Python 依赖**

```
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich \
    hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 \
    roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru \
    optree fvcore black pycocotools tensorboard huggingface_hub
```

### **5.4 安装 Detectron2（SAM3D 依赖）**

```
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps
```

### **5.5 可选安装：MoGe**

```
pip install git+https://github.com/microsoft/MoGe.git
```

------

# **6. HuggingFace 权限申请**

本项目依赖两个需要授权的模型：

- **SAM3**
   🔗 https://huggingface.co/facebook/sam3
- **SAM-3D-Objects**
   🔗 https://huggingface.co/facebook/sam-3d-objects

请在 HuggingFace 对应页面申请权限，并登录：

```
hf auth login
```

脚本会自动使用你的 Token。

------

# **7. 运行流程**

运行脚本前，请设置你的 Conda 激活脚本路径：

```
CONDA_SH="/your_path/miniconda3/etc/profile.d/conda.sh"
```

------

## **阶段 1：Qwen3-VL + SAM3 生成 2D Mask**

执行：

```
bash run_agent_with_vllm.sh
```

此脚本会：

1. 激活 `vllm` 环境
2. 启动 vLLM 服务，加载 Qwen3-VL
3. 激活 `sam3` 环境
4. 运行 `pipeline/run_sam3_agent_full.py`
5. 生成多物体 mask

输出目录：

```
outputs/master_with_vllm/masks/
```

------

## **阶段 2：SAM-3D-Objects 重建 3D Gaussian**

执行：

```
bash run_sam3d_from_masks.sh
```

此脚本会：

1. 激活 `sam3d-objects` 环境
2. 确保 SAM-3D-Objects 的 checkpoint 下载完成
3. 加载 RGB + masks
4. 生成每个物体的 `.pt` 文件
5. 重建并导出 3D Gaussian (`.ply`, `.gif`)

输出目录：

```
sam-3d-objects/outputs/torch_save_pt/
sam-3d-objects/gaussians/multi/
```

------

## **可选：一键式全流程执行**

```
bash run_pipeline.sh
```

该脚本会自动完成阶段 1 + 阶段 2。

------

# **Q&A**

## **Q1：下载模型时报 “Consistency check failed”？**

**原因：** 下载中断导致 HuggingFace 缓存中出现损坏的模型分片。
 **解决：删除损坏缓存并重新下载。**

```
rm -rf sam-3d-objects/checkpoints/hf
rm -rf ~/.cache/huggingface/hub   # 可选
bash run_sam3d_from_masks.sh
```

若要强制重新下载，可使用：

```
force_download=True
```

## **关于坐标系说明（PLY 输出方向）**

通过 **SAM-3D-Objects** 导出的 3D Gaussian `.ply` 文件默认处于 **相机坐标系** 下，其中：

- **+Z 轴** 为相机前向
- **+X 轴** 指向右侧
- **+Y 轴** 指向下方（典型计算机视觉坐标系）

因此，重建的对象是以 **相机前向 Z 轴** 对齐的，而不是世界坐标系。

如果需要将 `.ply` 放置到全局 **世界坐标系** 中（例如仿真器、机器人场景、NeRF / COLMAP world frame），必须执行一次 **相机 → 世界坐标系转换**：
$$
\mathbf{X}_{world} = \mathbf{R}_{c2w}\ \mathbf{X}_{camera} \ + \ \mathbf{t}_{c2w}
$$
其中：

- $\mathbf{R}_{c2w}$：相机到世界的旋转矩阵
- $\mathbf{t}_{c2w}$：相机到世界的平移向量
- $\mathbf{X}_{camera}$：高斯中心的相机系坐标
- $\mathbf{X}_{world}$：转换后的世界系坐标

完成转换后，你即可将 `.ply` 与全局场景或机器人环境正确对齐。
------

# **引用（Citation）**

### **SAM3**

```
@article{kirillov2024sam3,
  title={SAM 3: Segment Anything in Images and Videos},
  author={Kirillov, Alexander and Ravi, Nikhila and Mao, Weiyao and others},
  year={2024},
  url={https://github.com/facebookresearch/sam3}
}
```

### **SAM-3D-Objects**

```
@article{wu2024sam3dobjects,
  title={SAM-3D-Objects: Segment Anything in 3D Using 2D Masks},
  author={Wu, Yu and Mao, Weiyao and Kirillov, Alexander and others},
  year={2024},
  url={https://github.com/facebookresearch/sam3d}
}
```

------

# **致谢（Acknowledgements）**

本项目基于以下官方实现构建：

- **SAM3**
   GitHub: https://github.com/facebookresearch/sam3
   HuggingFace: https://huggingface.co/facebook/sam3
- **SAM-3D-Objects**
   GitHub: https://github.com/facebookresearch/sam3d
   HuggingFace: https://huggingface.co/facebook/sam-3d-objects

感谢原作者开放其卓越的研究成果与代码，使本流水线得以实现。