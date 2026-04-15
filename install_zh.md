# 50系卡安装SAM-3D / AnySplat   (torch2.7.0+cu128)

# 改动之处：

sam-3d-objects/pyproject.toml：
```
-PIP_EXTRA_INDEX_URL = "https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"  

改为 

+PIP_EXTRA_INDEX_URL = "https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu128"
```
requirements.inference.txt：
```
kaolin==0.17.0 改为 kaolin==0.18.0
```
requirements.txt：
```
nvidia-pyindex==1.0.9 改为 # nvidia-pyindex==1.0.9    （即注释掉）

torchaudio==2.5.1+cu121 改为 torchaudio, 
xformers==0.0.28.post3 改为 xformers （即取消指定torchaudio和xformers的版本）
```
requirements.p3d.txt：
```
tflash_attn==2.8.3 改为 flash_attn==2.7.3
```

# 运行以下安装命令 

```
uv venv --python 3.11

source .venv/bin/activate

uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

uv pip install -r AnySplat/requirements.txt --no-build-isolation

export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html"

uv pip install hatch-requirements-txt editables wheel

uv pip install -e './Sam-3d-objects[dev]' 

uv pip install -e './Sam-3d-objects[p3d]' --no-build-isolation 

uv pip install -e "./Sam-3d-objects[inference]"     --no-build-isolation     --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html 

uv pip install --index-strategy unsafe-best-match \
    "transformers>=4.48.3" \
    "iopaint>=1.2.0" \
    "numpy<2.0" \
    "opencv-python>=4.8.0" \
    "pyyaml>=6.0" \
    "requests>=2.31.0" \
    "tqdm>=4.66.0" \
    "setuptools"

uv pip install --index-strategy unsafe-best-match "git+https://github.com/facebookresearch/sam2.git"
```


## 解决运行AnySplat时的`Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead`的问题：
```
cd AnySplat/src/model/encoder/backbone/croco/curope/
```
把 kernels.cu中的

```
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type(), "rope_2d_cuda", ([&] {
改为
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.scalar_type(), "rope_2d_cuda", ([&] {
```
然后执行
```
python setup.py build_ext --inplace
```
