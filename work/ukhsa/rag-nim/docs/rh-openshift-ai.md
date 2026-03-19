# Using NVIDIA NIM Operator with Red Hat OpenShift AI (RHOAI)

This guide explains how to leverage the **NVIDIA NIM Operator** to serve models within a Red Hat OpenShift AI environment for the NVIDIA RAG Blueprint.

## 1. Prerequisites

*   **OpenShift Cluster** with the **NVIDIA GPU Operator** and **Node Feature Discovery (NFD) Operator** installed.
*   **NVIDIA NIM Operator** installed on the cluster.
*   **Red Hat OpenShift AI** (formerly RHODS) installed.

## 2. Model Serving via RHOAI Dashboard

While the NIM Operator can be used via raw Kubernetes manifests (as shown in the Helm chart), you can also integrate it with the RHOAI dashboard.

### Step A: Enable NVIDIA NIM Serving Runtime
1. In the OpenShift AI dashboard, go to **Settings** > **Serving runtimes**.
2. Ensure the **NVIDIA NIM** runtime is enabled. If not, click **Add serving runtime** and select the NVIDIA NIM template.

### Step B: Create a Data Science Project
1. Create or select a **Data Science Project**.
2. Navigate to the **Models and model servers** section.
3. Click **Add model server** and select **NVIDIA NIM** as the serving runtime.

### Step C: Deploy a Model
1. Click **Deploy model**.
2. Provide a name (e.g., `llm-nim`).
3. Select the model from the NVIDIA NGC catalog or provide a path to an OCI image.
4. Configure the number of replicas and GPU resources.

## 3. Connecting the RAG Blueprint

Once your models are deployed via the operator (either via RHOAI UI or Helm), you need to tell the RAG server where to find them.

### Using Helm
Use the `values-nim-operator.yaml` file provided in this repository to automatically point the RAG components to the operator-managed services.

```bash
helm install rag-server deploy/helm/nvidia-blueprint-rag \
  -f deploy/helm/values-nim-operator.yaml
```

### Internal Service URLs
The NIM Operator typically creates services using the pattern `http://<service-name>.<namespace>.svc.cluster.local:8000/v1`. 

Ensure your `APP_LLM_SERVERURL`, `APP_EMBEDDINGS_SERVERURL`, and `APP_RANKING_SERVERURL` environment variables match these internal DNS names.

## 4. Troubleshooting in OpenShift

*   **Security Context Constraints (SCC)**: Ensure the NIM pods have the appropriate SCCs (usually `nvidia-gpu-scc`) to access the GPUs.
*   **Image Pull Secrets**: Ensure your `ngc-secret` is present in the namespace where the NIMs are deployed.
*   **Storage**: Verify that the `NIMCache` can successfully provision a PVC for model weights.
