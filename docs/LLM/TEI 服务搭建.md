# TEI 服务搭建

TEI (Text Embeddings Inference) 是一个由 Hugging Face (HF) 提供的用于高效进行文本嵌入（text embeddings）推理的服务。它可以将输入的文本转化为向量（embedding），这些向量可以用于文本相似度搜索、语义搜索、信息检索等任务。

HF 提供的 **`text-embeddings-inference`** Docker 镜像就可以在多种硬件环境下（如 CPU 或 GPU）运行，支持不同文本模型的嵌入生成。

HF 提供[快捷部署的说明](https://huggingface.co/docs/text-embeddings-inference/quick_tour)，其中包含了镜像的快捷部署，和模型的快捷调用方式，值得说明的是，TEI 部署的 embedding 模型可以通过 OpenAI 库的 `openai.Embedding.create` 方法调用，一定程度上保证了项目的一致性。但是 rerank 服务的调用，只有访问 `/rerank` 接口，这是一个遗憾。

## Docker 镜像选择

Hugging Face TEI 官方提供两类镜像：

| 模式  | 镜像                                                       |
| --- | -------------------------------------------------------- |
| CPU | `ghcr.io/huggingface/text-embeddings-inference:cpu-1.9`  |
| GPU | `ghcr.io/huggingface/text-embeddings-inference:cuda-1.9` |

## 注意事项

如果模型私有，需要传入 `HF_TOKEN` 环境变量：

```bash
docker run -e HF_TOKEN=你的hf_token ...
```

## 部署 CPU 模式的 Embedding 服务

部署 CPU 模式的 TEI 服务，需要使用 CPU 专用镜像：

```bash
docker run -d \
  -p 8081:80 \
  -v ~/tei_models/data:/data \
  --name tei_cpu \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.9 \
  --model-id jinaai/jina-embeddings-v2-base-zh
```

## 部署 GPU 模式的 Embedding 服务

部署 GPU 模式的 TEI 服务，需要使用 GPU 专用镜像：

```bash
docker run --gpus all -d \
  -p 8082:80 \
  -v ~/tei_models/data:/data \
  --name tei_cuda \
  ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
  --model-id jinaai/jina-embeddings-v2-base-zh
```

参数说明：

- `--gpus all`：使用所有可用 GPU

## Docker Compose 示例

下面的配置可以同时管理 CPU 和 GPU 两个服务：

```yaml
services:
  tei_cpu:
    image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.9
    container_name: tei_cpu
    ports:
      - "8081:80"
    volumes:
      - ./data:/data
    restart: unless-stopped
    command: ["--model-id", "jinaai/jina-embeddings-v2-base-zh"]

  tei_cuda:
    image: ghcr.io/huggingface/text-embeddings-inference:cuda-1.9
    container_name: tei_cuda
    ports:
      - "8082:80"
    volumes:
      - ./data:/data
    restart: unless-stopped
    command: ["--model-id", "jinaai/jina-embeddings-v2-base-zh"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 测试接口

```bash
curl -X POST "http://localhost:8081/embeddings" \
     -H "Content-Type: application/json" \
     -d '{"inputs":["你好，世界"]}'
```

## 部署 CPU 模式的 Rerank 服务

部署 CPU 模式的 TEI 服务，需要使用 CPU 专用镜像：

```bash
docker run -d \
  -p 8083:80 \
  -v ~/tei_models/data:/data \
  --name tei_rerank_cpu \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.9 \
  --model-id BAAI/bge-reranker-base
```

## 部署 GPU 模式的 Rerank 服务

部署 GPU 模式的 TEI 服务，需要使用 GPU 专用镜像：

```bash
docker run --gpus all -d \
  -p 8084:80 \
  -v ~/tei_models/data:/data \
  --name tei_rerank_cuda \
  ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
  --model-id BAAI/bge-reranker-base
```

参数说明：

- `--gpus all`：使用所有可用 GPU

## Docker Compose 示例

下面的配置可以同时管理 CPU 和 GPU 两个服务：

```yaml
services:
  tei_cpu:
    image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.9
    container_name: tei_rerank_cpu
    ports:
      - "8083:80"
    volumes:
      - ./data:/data
    restart: unless-stopped
    command: ["--model-id", "BAAI/bge-reranker-base"]

  tei_cuda:
    image: ghcr.io/huggingface/text-embeddings-inference:cuda-1.9
    container_name: tei_rerank_cuda
    ports:
      - "8084:80"
    volumes:
      - ./data:/data
    restart: unless-stopped
    command: ["--model-id", "BAAI/bge-reranker-base"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 测试接口

```bash
curl -X POST "http://localhost:8083/rerank" \
     -H "Content-Type: application/json" \
     -d '{
			"query": "人工智能在医疗的应用",  
			"corpus": [  
			"AI在医学影像中的应用",  
			"Python后端开发最佳实践",  
			"机器学习模型训练"  
			]
		}'
```

