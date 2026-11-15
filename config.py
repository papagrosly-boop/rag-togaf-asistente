"""Configuración compartida. Un solo lugar para modelo, rutas y la frase de 'sin información'."""
import os
from dotenv import load_dotenv

load_dotenv()
# La clave de Groq se lee de GROQ_API_KEY (.env local o secrets de Streamlit)

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")   # Groq. Se puede cambiar sin editar: $env:LLM_MODEL="qwen/qwen3.8-27b"
PDF_FOLDER = "pdfs"
PERSIST_DIRECTORY = "chroma_db_multi"    # índice NUEVO (multilingüe). El viejo 'chroma_db' queda para la línea base
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# Frase exacta cuando la base no tiene la respuesta (el Avance 3 pide mantenerla exacta)
SIN_INFO = "No encontré esa información en los documentos de la base de conocimientos."


def get_llm(model=None, temperature=0):
    """LLM de Groq. Un solo lugar para cambiar de modelo o de proveedor."""
    from langchain_groq import ChatGroq
    modelo = model or LLM_MODEL
    extra = {"reasoning_effort": "low"} if "gpt-oss" in modelo else {}   # menos tokens de "razonamiento"
    return ChatGroq(model=modelo, temperature=temperature, max_retries=4, timeout=120, **extra)


def get_embeddings():
    """Embeddings multilingües (español <-> inglés). E5 exige prefijos 'query:' y 'passage:'."""
    from langchain_huggingface import HuggingFaceEmbeddings

    class E5Embeddings(HuggingFaceEmbeddings):
        def embed_documents(self, texts):
            return super().embed_documents([f"passage: {t}" for t in texts])

        def embed_query(self, text):
            return super().embed_query(f"query: {text}")

    return E5Embeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
