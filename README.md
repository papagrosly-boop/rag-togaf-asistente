# Asistente RAG - TOGAF 9.1

Sistema de recuperación y generación aumentada (RAG) desarrollado para consultar la documentación oficial de TOGAF 9.1, con trazabilidad de fuentes y evaluación automatizada mediante Ragas.

## Estructura del Proyecto

El repositorio está organizado de forma modular para separar la ingesta de datos, la lógica del motor RAG, la interfaz de usuario y la evaluación:

```text
rag_asistente/
├── pdfs/                     # Documentación oficial de TOGAF en PDF (Fase Preliminar, Introducción, Arquitectura de Negocio, Datos, Tecnología, etc.)
├── chroma_db/                # Base de datos vectorial local persistente
├── env/                      # Entorno virtual de Python
├── .env                      # Variables de entorno (API Key de Google)
├── ingest.py                 # Script de procesamiento, chunking y vectorización de los PDFs
├── rag_chain.py              # Lógica principal del RAG (recuperación top_k, prompt y conexión con Gemini)
├── app.py                    # Interfaz de chat interactiva (Streamlit)
├── evaluate_ragas.py         # Script de automatización de métricas con Ragas
├── test_dataset.json         # Dataset de evaluación (15 casos con ground truth y casos fuera de dominio)
├── ragas_evaluation_results.csv # Resultados exportados de la evaluación
└── requirements.txt          # Dependencias del proyecto
Flujo Técnico
Ingesta (ingest.py): Carga los documentos PDF desde la carpeta pdfs/, extrae el texto y aplica una estrategia de fragmentación (chunking) con tamaño y solapamiento controlados para no cortar definiciones clave del método ADM.

Vectorización y Base Vectorial: Los chunks se convierten a vectores mediante text-embedding-004 de Google y se almacenan localmente en ChromaDB junto con sus metadatos (origen y sección) para permitir la citación exacta.

Generación (rag_chain.py): Utiliza gemini-2.5-flash como modelo principal de respuestas, inyectando los fragmentos recuperados (top_k) en un System Prompt estricto que previene alucinaciones y maneja consultas fuera de alcance.

Interfaz (app.py): Aplicación web interactiva construida con Streamlit que mantiene el historial de la conversación, permite preguntas de seguimiento y muestra las fuentes utilizadas para cada respuesta.

Requisitos y Configuración
Clonar el repositorio:
git clone [https://github.com/tu-usuario/rag_asistente.git](https://github.com/tu-usuario/rag_asistente.git)
cd rag_asistente
Crear y activar el entorno virtual:
python -m venv env
# En Windows:
.\env\Scripts\activate
# En Linux/macOS:
source env/bin/activate
Instalar dependencias:
pip install -r requirements.txt
Configurar las credenciales:
Crea un archivo .env en la raíz del proyecto con tu clave de la API de Google:
GEMINI_API_KEY=tu_clave_aqui
Ejecución
Procesar documentos e inicializar la base vectorial:
python ingest.py
Ejecutar la evaluación de Ragas (métricas de fidelidad y relevancia):
python evaluate_ragas.py
Lanzar la interfaz de chat local:
streamlit run app.py
Despliegue
La aplicación se encuentra desplegada y accesible en línea:

URL Pública: https://rag-togaf-asistente-7hydy2mtegtek9tmrggba9.streamlit.app







