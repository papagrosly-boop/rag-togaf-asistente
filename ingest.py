import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PDF_FOLDER = "pdfs"
PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def index_academic_documents():
    if not os.path.exists(PDF_FOLDER) or not os.listdir(PDF_FOLDER):
        print(f"❌ Error: La carpeta '{PDF_FOLDER}' está vacía. Agrega tus PDFs académicos antes de continuar.")
        return None

    print("📄 1. Cargando documentos académicos de la carpeta 'pdfs'...")
    loader = PyPDFDirectoryLoader(PDF_FOLDER)
    raw_documents = loader.load()
    print(f"   ✓ Se cargaron {len(raw_documents)} páginas de documentos.")

    print("✂️ 2. Fragmentando texto (Chunking)...")
    # Justificación técnica para informe:
    # chunk_size=1000 y chunk_overlap=200 aseguran la captura de conceptos académicos complejos
    # manteniendo continuidad sintáctica entre fragmentos contiguos.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"   ✓ Se generaron {len(chunks)} chunks de texto.")

    print("🧬 3. Inicializando modelo de embeddings...")
    # Justificación técnica:
    # 'all-MiniLM-L6-v2' ofrece una excelente representación semántica en 384 dimensiones
    # con rendimiento óptimo y bajo consumo de recursos.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("💾 4. Indexando y guardando en la base de datos vectorial (ChromaDB)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"✅ Proceso finalizado con éxito. Base de conocimiento guardada en '{PERSIST_DIRECTORY}'.")
    return vectorstore

if __name__ == "__main__":
    index_academic_documents()