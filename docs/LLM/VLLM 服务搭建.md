# VLLM 服务搭建
## 安装 vLLM 应用

vLLM 官方推荐使用 uv 建新环境，为了和系统 python 环境区分，这里新建一个 conda 环境，使用 python 3.12。

```bash
conda create -n vllm python=3.12 -y
```

然后在 conda 环境中安装 vLLM，因为当前我的 GPU 版本是 CUDA 12.8，因此显示指定了 torch 的 cuda 版本是 cu128：

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
python -m pip install -U uv
uv pip install vllm --torch-backend=cu128
```

安装完成后，使用一个 python 脚本测试安装情况：

```bash
(vllm) ~$ python -c "import sys, torch, vllm; print(sys.executable); print(vllm.__version__); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
/home/brian/miniforge3/envs/vllm/bin/python
0.19.0
2.10.0+cu128
12.8
True
```
### 安装基础运行环境

在使用 vLLM 拉起大模型前， 需要安装 vLLm 的两个基础依赖，构建工具和 `duda-toolkit`：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install build-essential -y
sudo apt install cuda-toolkit-12-8 -y
sudo reboot
```

重启后，需要将 `cuda` 的路径添加到环境变量中：

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export CUDA_PATH=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' >> ~/.bashrc
source ~/.bashrc
```

然后使用下面的命令验证：

```bash
which nvcc
nvcc --version
ls -l /usr/local/cuda-12.8
```
 
## 启动 vLLM 服务
### 支持的模型列表

首先在使用 vLLM 拉起大模型前，需要先知道 vLLM 都支持哪些大模型，官方给了一个支持的[模型列表](https://docs.vllm.ai/en/stable/models/supported_models/)，在列表中可以看到 vLLM 原生支持的架构，这里以 `Qwen3/Qwen3-8B` 模型为例，支持列表中的状态如下：

| Architecture       | Models | Example HF Models     | [LoRA](https://docs.vllm.ai/en/stable/features/lora/) | [PP](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) |
| ------------------ | ------ | --------------------- | ----------------------------------------------------- | ----------------------------------------------------------------- |
| `Qwen3ForCausalLM` | Qwen3  | `Qwen/Qwen3-8B`, etc. | ✅︎                                                    | ✅︎                                                                |

这表示支持 `Qwen3ForCausalLM` 架构的模型，例如 HF 中的 `Qwen/Qwen3-8B` ，etc 表示同属 `Qwen3ForCausalLM` 架构的其他 Qwen3 模型也在这个支持范围内，例如：`Qwen/Qwen3-1.7B`、`Qwen/Qwen3-4B`。同时支持在 vLLM 里挂 `LoRA adapter`，以及 `Pipeline Parallelism` 多卡/多机部署。

实际部署时，根据应用场景选择使用基础模型，还是 `Instruct` 模型，对于同一个模型而言，他们有以下区别：

- `Qwen/Qwen3-4B`：基础续写模型，经过预训练专门用于续写的基础模型。
- `Qwen/Qwen3-4B-Instruct-2507`：对基础模型进行指令微调，通用指令执行模型。

### 常用参数说明

在官方的 `serve` [参数介绍](https://docs.vllm.ai/en/latest/cli/serve/)中，vLLM 拉起本地大模型服务支持很多参数，以下是最值得盯紧的参数：

- `MODEL` 或 `--model`：要加载的模型名或本地路径，比如 `Qwen/Qwen3-4B-Instruct-2507`。这是“跑哪个模型”。
- `--dtype`：模型权重/激活精度，常见是 `auto`、`half`、`bfloat16`。精度越高越稳，但越吃显存。
- `--quantization`：量化方式，比如 `bitsandbytes`、`gptq`、`awq`。显存不够时很关键。已有量化模型很多时候可自动识别，在线 `4-bit` 常见是显式写 `bitsandbytes`。
- `--max-model-len`：上下文长度上限。这个对显存影响非常大，往往比你想象得还大。
- `--max-num-seqs`：每一轮调度里，vLLM 最多同时处理多少条“序列”。与 `--max-model-len` 一起影响显存，显存不够时，将最大并发条数设小会很有用。
- `--gpu-memory-utilization`：vLLM 允许自己最多吃掉多少比例的 GPU 显存，默认 0.9。单卡桌面环境常用 0.80 到 0.85。
- `--kv-cache-dtype`：KV Cache 的精度，常见 `auto`、`float16`、`bfloat16`、`fp8`。显存紧张时，fp8 很有价值。
- `--cpu-offload-gb`：把一部分权重放到 CPU 内存，像“虚拟扩显存”。能帮更大模型起服务，但延迟会变差。
- `--host` / `--port`：服务监听地址和端口。127.0.0.1 只本机访问，0.0.0.0 局域网可访问。
- `--api-key`：给本地服务加访问密钥。
- `--served-model-name`：对外 API 返回的模型名，可以比 HF 名称更短。
- `--download-dir`：模型下载缓存目录。不写就走 Hugging Face 默认缓存。
- `--hf-token`：拉 gated/private 模型时必需，比如部分 Llama。
- `--trust-remote-code`：某些 HF 模型需要执行仓库里的自定义代码才能加载；开了兼容性更好，但有安全风险。
- `--revision`：固定模型版本，避免以后自动拉到新版本。
- `--load-format`：权重格式，常见 `auto`、`safetensors`、`gguf`。绝大多数 HF 模型建议 `auto` 或 `safetensors`。

### 参数的深度理解

下面这组理解方式，比较适合在本地部署时快速判断“显存为什么不够”：

- `dtype`、`quantization` 主要影响“模型本体”
- `kv-cache-dtype`、`max-model-len` 主要影响“上下文缓存”
- `gpu-memory-utilization` 主要影响“vLLM 最多能吃掉多少显存”
- `cpu-offload-gb` 主要影响“是否借用系统内存来缓解显存压力”

#### 显存的内容

当你用 `vLLM` 拉起一个模型时，显存里通常要放下几类内容：

- 模型权重
- 推理过程中使用的激活和中间计算结果
- `KV Cache`
- 框架自身开销，以及桌面环境、驱动和显示占用

所以本地部署是否成功，通常不是只看“模型参数量”，而是要同时看：

- 模型本体有多大
- 上下文缓存开得有多长
- 给 `vLLM` 留了多少显存预算

#### `dtype`

`dtype` 是“模型权重和激活”的数据类型。

可以这样理解：

- 数据类型越小，通常越省显存
- 数据类型越大，通常数值表示越宽裕
- 推理时不一定“越大越值得”，而是要在显存、速度和数值稳定性之间做平衡

常见值：

- `auto`
- `half` / `float16`
- `bfloat16`
- `float32`

如果没有启用量化，`dtype` 会直接影响权重的表示精度；如果已经启用量化，权重大小主要由 `quantization` 决定，但 `dtype` 仍会影响激活和部分未量化路径。

#### `quantization`

`quantization` 的核心作用，是用更低比特表示模型权重，从而降低模型本体的显存占用。

常见方法：

- `bitsandbytes`
- `gptq`
- `awq`

可以把它理解成：

- 不量化时，模型权重通常按 `fp16` / `bf16` 一类精度保存和加载
- 量化后，权重会以更低比特形式表示
- 这样能让更大的模型在较小显存上完成部署

量化的代价通常是：

- 可能出现一定精度损失
- 不同量化方法之间兼容性不同
- 某些情况下会带来额外的反量化开销

但在低显存显卡上，量化往往不是“锦上添花”，而是“能不能跑起来”的前提。

#### `kv-cache-dtype`

`kv-cache-dtype` 主要控制 `KV Cache` 的存储精度。

它直接影响：

- 上下文缓存占用多少显存
- 能存多少 `token`
- 能支持多长上下文
- 在固定显存下还能扛多少并发

可以这样记：

- `dtype` 更偏模型本体
- `kv-cache-dtype` 更偏上下文缓存

对于显存紧张的场景，较小的 `kv-cache-dtype` 往往很有帮助，例如 `fp8`。代价通常是可能有一点数值误差，但在工程上经常是值得的交换。
#### `max-model-len`

`max-model-len` 决定服务需要支持的最大上下文能力，也就是“单次请求最多能带多长的输入和输出总长度”。

它主要影响：

- 单次请求允许多长
- `KV Cache` 需要预留多大的能力
- 在固定显存下还能剩多少空间给并发和批处理

很重要的一点是：

- 这个值并不代表每次请求都会真正用满
- 但 `vLLM` 必须按这个上限来规划资源

因此：

- 值越大，显存压力通常越大
- 值越大，在固定显存下能容纳的并发通常越少
- 值设置过大时，服务甚至可能根本起不来
#### `gpu-memory-utilization`

`gpu-memory-utilization` 表示当前这一个 `vLLM` 实例最多可使用多少比例的 GPU 显存。

它的作用更像是：

- 给 `vLLM` 划预算
- 决定模型、缓存和运行时最多能占多大空间

这个参数和下面两项经常一起影响启动结果：

- `kv-cache-dtype`
- `max-model-len`

如果你在桌面环境里跑模型，通常不要把它设得太激进，因为系统显示、浏览器、IDE 和桌面合成都在占显存。

#### `cpu-offload-gb`

`cpu-offload-gb` 的作用，是把一部分模型权重放到系统内存里，以缓解显存压力。

这里的 `CPU RAM` 指的是操作系统可用的系统内存，也就是主板上的内存条，不是显卡显存。

更准确地说：

- 它不是让 CPU 帮 GPU 分担主要计算
- 主要计算仍然发生在 GPU 上
- 它缓解的是“显存容量压力”，不是“GPU 算力压力”

开启后，可以把它理解成：

- 一部分权重不再常驻显存
- 这些权重保存在系统内存里
- 推理时通过 CPU-GPU 互连按需访问

这种方式的优点是：

- 模型更容易先启动成功
- 对“差一点点显存”的情况很有帮助

缺点也很明显：

- 延迟会更高
- 吞吐通常会下降
- 对 CPU-GPU 传输速度更敏感

所以从工程角度，`cpu-offload-gb` 更适合：

- 模型差一点点装不下
- 想先验证服务能不能起来
- 可以接受变慢

不太适合作为长期主力方案，尤其是在单卡低显存机器上。

#### 总结

如果你遇到“模型起不来”或“显存不够”，可以按下面顺序排查：

1. 先看是不是模型本体太大：
   这时优先考虑 `quantization`
2. 再看是不是上下文开太长：
   这时优先考虑 `max-model-len` 和 `kv-cache-dtype`
3. 如果还是差一点显存：
   再考虑 `cpu-offload-gb`

对低显存运行大模型，通常优先级是：

1. `quantization`
2. `kv-cache-dtype`
3. `max-model-len`
4. `cpu-offload-gb`

这是因为：

- `quantization` 让模型更容易常驻显存
- `kv-cache-dtype` 和 `max-model-len` 直接决定长上下文的缓存开销
- `cpu-offload-gb` 虽然能帮忙启动，但速度上的代价通常更明显

## 启动 vLLM 服务示例

当前我的环境是 Ubuntu 24.04 + Nvidia RTX 4060 8GB，在启动大模型服务时，由于 GPU 只有 8GB，因此不适合大量级的模型，在当前环境下，我使用 `--quantization` `4-bit` 的量化形式，需需要在 conda 环境中安装 `bitsandbytes`。

```bash
(vllm) ~$ python -m pip install "bitsandbytes>=0.49.2"
```

接下来启动一个大模型，当前的 8G 内存，可以跑较长上下文的 Qwen/Qwen3-4B-Instruct-2507：

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8080 \
  --served-model-name qwen3-4b \
  --quantization bitsandbytes \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --api-key my-local-key
```

服务启动后，可以使用下面的命令测试 vLLM 服务情况：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-local-key" \
  -d '{
    "model": "qwen3-4b",
    "messages": [
      {"role": "user", "content": "你好，请做个自我介绍"}
    ],
    "temperature": 0.6,
    "top_p": 0.95
  }'
```

对于 8B 参数量的模型，则只能选择更小的上下文长度，同时给予其更多的显存：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8080 \
  --served-model-name Llama-3.1-8B-Instruct \
  --quantization bitsandbytes \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 1
```

### 注意事项

如果你使用的是私有模型，请在环境变量中添加你的 `HF_TOKEN` 。
