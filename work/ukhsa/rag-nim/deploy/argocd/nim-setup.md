# Self-Hosted NVIDIA NIM Setup Instructions

To use the NVIDIA RAG Blueprint with self-hosted NIMs, follow these steps:

## 1. Deploy NIMs via Operator (Recommended)
The easiest way to deploy NIMs is using the **NVIDIA NIM Operator**. The provided Helm chart includes built-in support for the operator.

To deploy the NIMs alongside the RAG server, use the `values-nim-operator.yaml` override:

```bash
helm install rag-blueprint deploy/helm/nvidia-blueprint-rag \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml \
  -f deploy/helm/values-nim-operator.yaml
```

This will automatically create `NIMCache` and `NIMService` resources for the LLM, Embedding, and Reranking models. It also enables an **ArgoCD Pre-Sync Job** that uses **Protect AI ModelScan** to verify the integrity of the downloaded models before the services are started.

### Model Security Scanning (ModelScan)
The scanner ensures that models don't contain malicious code. You can configure it in `values.yaml`:

```yaml
modelScan:
  enabled: true
  failOnViolation: true       # Block deployment if vulnerabilities found
  severityThreshold: "high"  # Severity to trigger failure
```

### Manual Deployment
If you prefer to deploy NIMs manually, ensure they are reachable via these internal service names:
- **LLM**: `http://llm-nim:8000/v1`
- **Embedding**: `http://nemoretriever-embedding-ms:8000/v1`
- **Reranking**: `http://nemoretriever-ranking-ms:8000/v1`

## 2. Configure Argo CD
The provided `argocd-app.yaml` is configured to use the `values-nim.yaml` override. 

```bash
kubectl apply -f deploy/argocd/argocd-app.yaml
```

## 3. Verify Connectivity
Ensure that the RAG server pods can resolve and reach the NIM service endpoints. You can test this by exec-ing into a RAG server pod:

```bash
curl http://llm-nim:8000/v1/models
```

## 4. Model Names
Ensure the model names in `values-nim.yaml` match the models deployed in your NIMs.
- `APP_LLM_MODELNAME`: e.g., `nvidia/llama-3.3-nemotron-super-49b-v1.5`
- `APP_EMBEDDINGS_MODELNAME`: e.g., `nvidia/nv-embedqa-e5-v5`
- `APP_RANKING_MODELNAME`: e.g., `nvidia/nv-rerankqa-mistral-4b-v3`
