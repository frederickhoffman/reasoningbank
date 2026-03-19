import sys
import os
import logging

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "src"))

from unittest.mock import MagicMock
from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.llm import get_llm
from nvidia_rag.utils.embedding import get_embedding_model
from nvidia_rag.utils.reranker import _get_ranking_model, TEIRerank

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llm_factory():
    logger.info("Testing LLM Factory...")
    config = NvidiaRAGConfig()
    
    # Test vLLM
    config.llm.model_engine = "vllm"
    config.llm.server_url = "http://localhost:8000/v1"
    config.llm.model_name = "llama3"
    
    llm = get_llm(config=config, model="llama3")
    logger.info(f"vLLM type: {type(llm)}")
    assert "ChatOpenAI" in str(type(llm))
    assert llm.openai_api_base == "http://localhost:8000/v1"

    # Test NVIDIA (default)
    config.llm.model_engine = "nvidia-ai-endpoints"
    config.llm.server_url = "http://nim-llm:8000"
    llm = get_llm(config=config, model="nvidia/llama-3.3")
    logger.info(f"NVIDIA type: {type(llm)}")
    assert "ChatNVIDIA" in str(type(llm))

def test_embedding_factory():
    logger.info("Testing Embedding Factory...")
    config = NvidiaRAGConfig()
    
    # Test TEI
    config.embeddings.model_engine = "tei"
    config.embeddings.server_url = "http://tei:80"
    config.embeddings.model_name = "bge-large"
    
    embed = get_embedding_model(config=config)
    logger.info(f"TEI Embed type: {type(embed)}")
    assert "OpenAIEmbeddings" in str(type(embed))
    assert embed.openai_api_base == "http://tei:80"

def test_reranker_factory():
    logger.info("Testing Reranker Factory...")
    config = NvidiaRAGConfig()
    
    # Test TEI Reranker
    config.ranking.model_engine = "tei"
    config.ranking.server_url = "http://tei-rerank:80"
    config.ranking.model_name = "bge-reranker"
    
    reranker = _get_ranking_model(config=config, model="bge-reranker", url="http://tei-rerank:80")
    logger.info(f"TEI Reranker type: {type(reranker)}")
    assert isinstance(reranker, TEIRerank)
    assert reranker.endpoint_url == "http://tei-rerank:80"

if __name__ == "__main__":
    try:
        test_llm_factory()
        test_embedding_factory()
        test_reranker_factory()
        logger.info("All manual verification tests passed!")
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        sys.exit(1)
