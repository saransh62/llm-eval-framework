import pytest

from app.config import load_settings
from app.llm import FakeLLM
from app.rag import RAGPipeline, build_retriever


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def retriever(settings):
    return build_retriever(settings)


@pytest.fixture()
def pipeline(retriever):
    return RAGPipeline(retriever, FakeLLM(), top_k=4)
