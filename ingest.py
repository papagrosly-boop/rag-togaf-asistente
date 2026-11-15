"""Fase 1: reindexa los PDFs con embeddings multilingües y metadatos limpios (documento + página)."""
import os
import re
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import PDF_FOLDER, PERSIST_DIRECTORY, get_embeddings


def nombre_documento(ruta: str) -> str:
    base = os.path.basename(ruta)
    base = re.sub(r"\s*\(\d+\)", "", base)       # quita ' (1)' de los nombres duplicados
    return os.path.splitext(base)[0]


def index_academic_documents():
    if not os.path.isdir(PDF_FOLDER) or not os.listdir(PDF_FOLDER):
        print(f"Error: la carpeta '{PDF_FOLDER}' está vacía.")
        return None

    print("1. Cargando PDFs...")
    paginas = PyPDFDirectoryLoader(PDF_FOLDER).load()
    print(f"   {len(paginas)} páginas cargadas.")

    # Metadatos limpios: lo único que necesitamos para citar
    for d in paginas:
        d.metadata = {
            "documento": nombre_documento(d.metadata.get("source", "")),
            "pagina": int(d.metadata.get("page", 0)) + 1,   # PyPDF cuenta desde 0
            "source": d.metadata.get("source", ""),
        }

    print("2. Fragmentando (chunk 1000 / overlap 200)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(paginas)
    print(f"   {len(chunks)} chunks.")

    print("3. Cargando embeddings multilingües (la primera vez descarga ~470 MB)...")
    embeddings = get_embeddings()

    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)   # evita duplicados si reindexas

    print(f"4. Indexando en '{PERSIST_DIRECTORY}'...")
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIRECTORY)

    print("\nPrueba rápida (pregunta en español sobre texto en inglés):")
    for q in ["¿Cuáles son los objetivos de la Fase Preliminar?", "¿Qué es el Enterprise Continuum?"]:
        print(f"  > {q}")
        for d in vs.similarity_search(q, k=3):
            print(f"     - {d.metadata['documento']}, pág. {d.metadata['pagina']}")
    print("\nListo.")
    return vs


if __name__ == "__main__":
    index_academic_documents()
