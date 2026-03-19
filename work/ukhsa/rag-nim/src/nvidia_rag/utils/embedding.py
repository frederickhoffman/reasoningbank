# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The wrapper for interacting with embedding models.
1. get_embedding_model: Get the embedding model. Uses the NVIDIA AI Endpoints or HuggingFace.
"""

import logging
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from nvidia_rag.utils.common import NVIDIA_API_DEFAULT_HEADERS, sanitize_nim_url
from nvidia_rag.utils.configuration import NvidiaRAGConfig

logger = logging.getLogger(__name__)


def get_embedding_model(
    model: str,
    url: str,
    config: NvidiaRAGConfig | None = None,
    truncate: str | None = "END",
) -> Embeddings:
    """Create the embedding model.

    Args:
        model: Model name
        url: URL endpoint
        config: NvidiaRAGConfig instance. If None, creates a new one.
        truncate: Truncation strategy
    """
    if config is None:
        config = NvidiaRAGConfig()

    # Sanitize the URL
    url = sanitize_nim_url(url, model, "embedding")

    api_key = config.embeddings.get_api_key()

    if url:
        logger.info("Using self-hosted embedding NIM at %s", url)
        return NVIDIAEmbeddings(
            base_url=url,
            model=model,
            api_key=api_key,
            truncate=truncate or "END",
            dimensions=config.embeddings.dimensions,
            default_headers=NVIDIA_API_DEFAULT_HEADERS,
        )

    logger.info("Using embedding model %s from NVIDIA API Catalog", model)
    return NVIDIAEmbeddings(
        model=model,
        api_key=api_key,
        truncate=truncate or "END",
        dimensions=config.embeddings.dimensions,
        default_headers=NVIDIA_API_DEFAULT_HEADERS,
    )

    raise RuntimeError(
        "Unable to find any supported embedding model. Supported engine is huggingface and nvidia-ai-endpoints."
    )
