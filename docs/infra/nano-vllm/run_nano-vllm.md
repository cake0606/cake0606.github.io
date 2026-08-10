# 用 `uv` 跑 `nano-vllm`

## 先确认前提

`nano-vllm` 不是纯 CPU 项目，代码里直接用了：

- `torch.cuda`
- `torch.distributed` 的 `nccl`
- `flash-attn`
- `triton`

所以默认前提是：

- Linux
- NVIDIA GPU
- CUDA 环境可用
- Linux 侧安装了 CUDA toolkit，并且 `nvcc` 可用
- `CUDA_HOME` 已指向 Linux 的 CUDA 安装目录
- Python 版本是 `3.10` 到 `3.12`

注意：你当前文档仓库 `cakeman.github.io` 的 `pyproject.toml` 里是 `requires-python = ">=3.13"`，但 `nano-vllm/pyproject.toml` 要求 `>=3.10,<3.13`。  
这意味着不要复用文档仓库的 Python 环境，应该单独给 `nano-vllm` 建一个 `uv` 虚拟环境。

## 1. 安装 `uv`

如果你本机还没有 `uv`，先装它：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

装完后重新开一个 shell，确认：

```bash
uv --version
```

## 2. 进入项目目录

```bash
cd /home/zzz/project/nano-vllm
```

## 3. 用 `uv` 创建独立环境

推荐直接用 Python 3.12：

```bash
uv venv --python 3.12
source .venv/bin/activate
```

如果你机器上还没有 Python 3.12，可以先：

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```

## 4. 安装项目依赖

建议先安装 `torch`，再安装项目本身：

```bash
uv pip install torch
uv pip install -e .
```

之所以先装 `torch`，是因为 `flash-attn` 在构建时会导入 `torch`，但它自己的打包元数据没有把这件事声明完整。  
我已经在项目的 `pyproject.toml` 里补了：

```bash
[tool.uv.extra-build-dependencies]
flash-attn = ["torch"]
```

这一步会拉起这些依赖：

- `torch`
- `triton`
- `transformers`
- `flash-attn`
- `xxhash`

如果你在这一步还是卡在 `flash-attn`，通常不是 `uv` 的问题，而是：

- CUDA 版本和 PyTorch 不匹配
- 本机没有可用的编译环境
- Linux 环境里没有 `nvcc`
- `CUDA_HOME` 没有设置
- GPU 架构 / 驱动不满足要求

这时建议先单独确认：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
which nvcc
echo $CUDA_HOME
```

如果这里 `False`，先不要继续跑 `nano-vllm`，先把 PyTorch + CUDA 打通。

如果 `torch.cuda.is_available()` 是 `True`，但 `which nvcc` 没结果，说明你只是有驱动或运行时，还没有 Linux 侧的 CUDA 编译工具链。  
这种情况下 `flash-attn` 仍然会安装失败。

## 5. 下载模型

项目的示例代码默认读取这个目录：

```bash
~/huggingface/Qwen3-0.6B/
```

先下载：

```bash
hf download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/
```

如果你还没有 Hugging Face 的新 CLI，先装：

```bash
uv pip install -U huggingface_hub
```

确认命令可用：

```bash
hf version
```

如果模型需要登录权限，再执行：

```bash
hf auth login
```

## 6. 跑官方示例

仓库里已经有可直接运行的示例：

```bash
python example.py
```

这个文件里实际做的是：

- 从 `~/huggingface/Qwen3-0.6B/` 读取模型
- 用 `AutoTokenizer` 构造 chat prompt
- 用 `LLM(..., enforce_eager=True, tensor_parallel_size=1)` 启动
- 输出生成结果

第一次跑建议保留 `enforce_eager=True`，先不要急着开更激进的优化，这样更容易排查环境问题。

## 7. 最小自检

如果你只想确认环境是否通了，可以直接跑：

```bash
python -c "from nanovllm import LLM, SamplingParams; print('import ok')"
```

再检查 GPU：

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
```

只要这两步都正常，再跑 `python example.py`。

## 8. 你现在最该注意的坑

### Python 版本坑

不要在 `cakeman.github.io` 的 `3.13` 环境里装 `nano-vllm`。  
这个项目需要 `<3.13`。

### GPU / NCCL 坑

`nanovllm/engine/model_runner.py` 里初始化了：

```python
dist.init_process_group("nccl", ...)
torch.cuda.set_device(rank)
torch.set_default_device("cuda")
```

所以没有 NVIDIA GPU、CUDA 或 NCCL 不通的话，程序基本起不来。

另外，`flash-attn` 安装阶段还需要 `nvcc`。  
也就是说，“PyTorch 能看到 GPU” 不等于 “已经具备编译 `flash-attn` 的条件”。

### 模型路径坑

`example.py` 里路径写死成了：

```python
~/huggingface/Qwen3-0.6B/
```

如果你把模型下到了别处，要改这个变量：

```python
path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
```

## 一套最顺的命令

```bash
cd /home/zzz/project/nano-vllm
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch
uv pip install -e .
uv pip install -U huggingface_hub
hf download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/
python example.py
```

## 如果你想继续

下一步通常有两种：

- 想先跑通：直接用 `example.py`
- 想看吞吐：再跑 `python bench.py`
