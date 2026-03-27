# 🦅 RAG-NIM: Secure GitOps for NVIDIA RAG on OpenShift

This repository provides a production-ready implementation of a Retrieval-Augmented Generation (RAG) system, automated via **GitOps** and secured with **proactive model scanning**. It is designed specifically for private **OpenShift** environments where data privacy and model integrity are paramount.

## 🏗️ System Design

The system orchestrates four key layers to provide a seamless RAG experience:

1.  **Orchestration (Argo CD)**: Automates the deployment and synchronization of the entire stack.
2.  **Inference (NVIDIA NIM Operator)**: Dynamically provisions and caches GPU-optimized AI models (LLMs, Embeddings, and Rerankers) directly within your OpenShift cluster.
3.  **Security (Protect AI ModelScan)**: A "security gate" that audits model weights for arbitrary code execution vulnerabilities before they are loaded.
4.  **Application (NVIDIA RAG Blueprint Server)**: A high-performance FastAPI server that manages document ingestion and retrieval pipelines.

---

```mermaid
graph TD
    Repo[Repository] --> Argo[ArgoCD]
    Argo --> NC[NIMCache]
    NC --> PV[Storage]
    PV --> Scan[ModelScan]
    Scan --> NS[NIMService]
    Argo --> RAG[RAG Server]
    Argo --> Ing[Ingestion]
    RAG --> NS
    Ing --> NS
    Ing --> Milvus[VectorDB]
    User[End User] --> RAG
```

### 🗝️ Key Components

1.  **NVIDIA NIM Operator**: Automates the lifecycle of AI models. It handles everything from downloading weights to optimized CUDA orchestration.
2.  **Argo CD**: Provides a declarative GitOps workflow. Your "Source of Truth" in Git is automatically reflected in your cluster.
3.  **Protect AI ModelScan**: A critical security layer that scans model weights for arbitrary code execution vulnerabilities *before* they are loaded into memory.
4.  **NVIDIA RAG Server**: A highly optimized FastAPI application that leverages LangChain and NVIDIA AI Endpoints for hybrid search and generation.

---

## 🌟 Features

*   **🔒 Air-Gapped Ready**: Designed to run entirely within your private network. No data leaves your cluster.
*   **🛡️ Secure-by-Design**: Integrated ModelScan ensures that external models don't introduce supply chain attacks.
*   **⚡ Optimized Performance**: Leverages TensorRT-LLM and CUDA via the NIM Operator for blazing-fast inference.
*   **🤖 Universal Model Support**: Easily swap between Llama-3, Nemotron, Mistral, and more using simple Helm overrides.

---

## 🛠️ Setup & Deployment

### 1. Cluster Access
Before deploying, you must connect your terminal to your target **OpenShift** cluster.

1.  **Find your API URL**: This is the cluster endpoint (e.g., `https://api.ocp-col.ukhsa.gov.uk:6443`).
2.  **Obtain your API Token**:
    *   Log in to the **OpenShift Web Console**.
    *   Click your **Name/Username** in the top-right corner.
    *   Select **Copy Login Command**, then click **Display Token**.
3.  **Log in to your terminal**:
    ```bash
    oc login --token=sha256~<YOUR_TOKEN> --server=<YOUR_API_URL>
    ```

> [!TIP]
> Once logged in, your terminal context will be saved. You only need to run this command once per session.

### 2. Prerequisites & CLI Tools
To deploy from your command line, you need the **OpenShift CLI (`oc`)** and **Helm** installed.

#### 🛠️ Install CLI Tools (Linux Portable)
If you do not have these tools, run these commands to install them to your local binary folder:

```bash
# 1. Install OpenShift CLI (oc)
curl -sLO https://mirror.openshift.com/pub/openshift-v4/clients/ocp/latest/openshift-client-linux.tar.gz
tar -xzf openshift-client-linux.tar.gz
sudo mv oc kubectl /usr/local/bin/

# 2. Install Helm
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

#### 📋 Cluster Requirements
*   **OpenShift Cluster** (v4.12+) with NVIDIA GPUs.
*   **NVIDIA GPU Operator** installed.
*   **NVIDIA NIM Operator** installed.
*   **Argo CD** (optional) for GitOps-based deployment.

### 3. Configuration & Secrets

#### 🛡️ Secure Local Environment
To manage your credentials safely without committing them to Git:
1.  **Copy the template**:
    ```bash
    cp .env.example .env
    ```
2.  **Edit `.env`**: Add your `NVIDIA_API_KEY`, `OPENSHIFT_API_URL`, and `OPENSHIFT_TOKEN`. This file is already in `.gitignore`.

#### 🔑 NVIDIA NGC Setup
The NIM Operator and RAG Server both use your **NGC API Key** for authorization.

1.  **Generate API Key**: Log in to [NGC](https://ngc.nvidia.com) > Setup > API Key.
2.  **Create Image Pull Secret**:
    ```bash
    oc create secret docker-registry ngc-secret \
      --docker-server=nvcr.io \
      --docker-username="\$oauthtoken" \
      --docker-password=<YOUR_NGC_API_KEY> \
      -n nvidia-rag
    ```
3.  **Create API Key Secret**:
    ```bash
    oc create secret opaque ngc-api \
      --from-literal=NGC_API_KEY=<YOUR_NGC_API_KEY> \
      -n nvidia-rag
    ```

#### 🛡️ OpenShift Security (SCC Binding)
If using **OpenShift**, you must bind the `nvidia-gpu-scc` to the application's service account to allow GPU access:

```bash
# Apply the declarative binding provided in this repo
oc apply -f deploy/openshift/scc-rolebinding.yaml
```

---

## 🚀 Deployment Options

### Option A: GitOps via Argo CD (Recommended)
This method automates the entire stack sync.
1.  Apply the application manifest:
    ```bash
    oc apply -f deploy/argocd/argocd-app.yaml
    ```
2.  In the Argo CD dashboard, locate `nvidia-rag-blueprint` and click **Sync**.

### Option B: Manual via Helm CLI
Use the environment-specific values files for a tailored deployment.

#### 📦 1. Add Helm Repositories
The chart depends on external services (NIM, Elastic, Prometheus). Note that the **NVIDIA NGC** repo requires authentication:

```bash
# Add NVIDIA repo (requires your NGC API Key)
helm repo add nemo-microservices https://helm.ngc.nvidia.com/nvidia/nemo-microservices \
  --username="\$oauthtoken" \
  --password=<YOUR_NGC_API_KEY>

# Add public repositories
helm repo add elastic https://helm.elastic.co
helm repo add zipkin https://zipkin.io/zipkin-helm
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

helm repo update
```

#### 🏗️ 2. Build Dependencies
Fetch the local sub-charts before installing:

```bash
helm dependency build deploy/helm/nvidia-blueprint-rag
```

#### 🚀 3. Execute Installation
If you have already manually created your `ngc-secret` and `ngc-api`, use the `--set` flags to skip secret creation:

```bash
helm install rag-blueprint deploy/helm/nvidia-blueprint-rag \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml \
  -f deploy/helm/values-nim-operator.yaml \
  -f deploy/helm/values-openshift.yaml \
  --set ngcApiSecret.create=false \
  --set imagePullSecret.create=false
```

When deploying with the **NVIDIA NIM Operator**, the system automatically creates `NIMCache` and `NIMService` resources. 

### Model Names
Ensure the model names in your overrides match your target hardware:
*   **LLM**: `nvidia/llama-3.3-nemotron-super-49b-v1.5`
*   **Embedding**: `nvidia/llama-3.2-nv-embedqa-1b-v2`
*   **Reranking**: `nvidia/llama-3.2-nv-rerankqa-1b-v2`

### Verification
Once deployed, verify that the RAG server can reach the internal NIM endpoints:
```bash
# Exec into a RAG server pod and test the NIM API
curl http://rag-server-nim-llm:8000/v1/models
```

---

## 🛡️ Security: ModelScan Integration

This blueprint includes an **ArgoCD Pre-Sync Security Hook**. Before any new model is synchronized, an audit Job runs **Protect AI ModelScan** to verify model weights for arbitrary code execution vulnerabilities.

---

## 📥 Document Ingestion Lifecycle

Ingestion converts raw files (PDFs, images) into searchable vector embeddings.

1.  **UI Upload**: Upload documents directly via the Chat UI.
2.  **Batch Ingestion**: Use the local [batch_ingestion.py](scripts/batch_ingestion.py) script to ingest large datasets from your local machine to the cluster.

---

## 💬 Interacting with the RAG System

1.  **Blueprint UI**: Responsive chat interface available at `http://<frontend-service-ip>:3000`.
2.  **API Documentation**:
    *   **RAG Server**: `http://<rag-server-ip>:8081/docs`
    *   **Ingestor**: `http://<ingestor-server-ip>:8082/docs`

---

## 🧪 Testing & Ingestion
Use this consolidated block to verify your deployment and start chatting:

```bash
# 1. Start Port-Forwarding (Background)
# This bridges the cluster services to your local machine
oc port-forward svc/rag-server 8081:8081 -n nvidia-rag & \
oc port-forward svc/ingestor-server 8082:8082 -n nvidia-rag & \
oc port-forward svc/rag-frontend 3000:3000 -n nvidia-rag &

# 2. Run the automated smoke test
# Wait for connections to stabilize, then run the verification script
# NOTE: On Windows, you may need to use 'python3' or 'py'
sleep 5
python3 tests/verify_deployment.py --rag-url http://localhost:8081 --ingest-url http://localhost:8082

# 3. Batch Ingest your UKHSA dataset
# Unzip and upload in bulk to the vector database
unzip data/dataset.zip -d data/my_dataset
python scripts/batch_ingestion.py \
  --folder data/my_dataset \
  --collection-name "ukhsa_collection" \
  --create_collection \
  --ingestor-port 8082

# 4. Access the UI
# Open your browser to: http://localhost:3000
```

> [!TIP]
> To stop all background port-forwarding sessions later, simply run `killall oc`.


