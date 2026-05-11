# Install on RTX 50-series GPUs (torch 2.7.0 + cu128, also works on 3090,4090)

One-command installer:

```
bash scripts/install_env.sh
```

This document is the manual step-by-step installation reference. Use it if you want to inspect or run each installation step yourself.


# Run the installation commands below

```
git submodule update --init --recursive

uv venv --python 3.11

source .venv/bin/activate

export PYTHONPATH="$(pwd)/submodule/Sam-3d-objects/notebook:$(pwd)/submodule/Sam-3d-objects:${PYTHONPATH:-}"

uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

uv pip install -r submodule/AnySplat/requirements.txt --no-build-isolation

export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html"

uv pip install hatch-requirements-txt editables wheel

uv pip install -e './submodule/Sam-3d-objects[dev]'

uv pip install -e './submodule/Sam-3d-objects[p3d]' --no-build-isolation

uv pip install -e "./submodule/Sam-3d-objects[inference]"     --no-build-isolation     --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html

uv pip install --index-strategy unsafe-best-match \
    "transformers>=4.48.3" \
    "iopaint>=1.2.0" \
    "numpy<2.0" \
    "opencv-python>=4.8.0" \
    "pyyaml>=6.0" \
    "requests>=2.31.0" \
    "tqdm>=4.66.0" \
    "setuptools" \
    "huggingface_hub" \
    "einops"

uv pip install --index-strategy unsafe-best-match "git+https://github.com/facebookresearch/sam3.git"
```

## SAM3 model access

`facebook/sam3` is a gated model on HuggingFace. Request access on the model page first, then log in:
```
huggingface-cli login
```


## Fix the AnySplat warning: `Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead`
```
cd submodule/AnySplat/src/model/encoder/backbone/croco/curope/
```
In `kernels.cu`, change:

```
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type(), "rope_2d_cuda", ([&] {
```

to:

```
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.scalar_type(), "rope_2d_cuda", ([&] {
```

Then run:
```
python setup.py build_ext --inplace
```


## Optional: extra dependencies for `pipeline/mesh2mjcf.py`

The mesh-to-MJCF converter (see README §5) does not require anything beyond the
Python standard library by default, but the following features each need an
extra package:

```
# Convex decomposition (--cd)
uv pip install coacd trimesh

# Preview viewer (--verbose)
uv pip install mujoco
```

`trimesh` is also used for multi-material OBJ splitting, but it is usually
already installed as part of the Sam-3d-objects extras above. Install it
explicitly only if `python -c "import trimesh"` fails.


# Completed changes compared to the original repository:

submodule/Sam-3d-objects/pyproject.toml:
```
-PIP_EXTRA_INDEX_URL = "https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"  

change to

+PIP_EXTRA_INDEX_URL = "https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu128"
```
requirements.inference.txt:
```
kaolin==0.17.0 change to kaolin==0.18.0
```
requirements.txt:
```
nvidia-pyindex==1.0.9 change to # nvidia-pyindex==1.0.9    (comment it out)

torchaudio==2.5.1+cu121 change to torchaudio,
xformers==0.0.28.post3 change to xformers    (remove the pinned torchaudio and xformers versions)
```
requirements.p3d.txt:
```
tflash_attn==2.8.3 change to flash_attn==2.7.3
```