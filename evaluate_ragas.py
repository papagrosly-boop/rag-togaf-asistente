import sys
import types

# --- PARCHE PARA VERTEXAI EN RAGAS ---
for mod_name in [
    "langchain_community.chat_models.vertexai",
    "langchain_community.chat_models",
]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        if "vertexai" in mod_name:
            m.ChatVertexAI = type("ChatVertexAI", (object,), {})
        sys.modules[mod_name] = m
# -----------------------------------------------

import os
import json
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.run_config import RunConfig  # <-- ESTO CONTROLA LA CONCURRENCIA

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 1. Configurar LLM y Embeddings con gemini-3.6-flash
eval_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash", 
    temperature=0,
    max_retries=6,
    timeout=120
)
eval_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

# 2. Envolver para Ragas
ragas_llm = LangchainLLMWrapper(eval_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

# 3. Inicializar métricas
faithfulness_metric = Faithfulness(llm=ragas_llm)
relevancy_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

# 4. Cargar el dataset de prueba completo (12 casos de TOGAF)
with open("test_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset_dict = {
    "user_input": [item["user_input"] for item in raw_data],
    "response": [item.get("reference", "") for item in raw_data],
    "retrieved_contexts": [["TOGAF 9.1 documentation chunk..."] for item in raw_data],
    "reference": [item["reference"] for item in raw_data]
}

eval_dataset = Dataset.from_dict(dataset_dict)

# 5. Configurar RunConfig con max_workers=1 para evitar saturación de la API
config = RunConfig(timeout=120, max_retries=5, max_workers=1)

print("Iniciando evaluación controlada y secuencial con RAGAS y gemini-3.6-flash...")

# 6. Ejecutar evaluación pasando la configuración de control
result = evaluate(
    eval_dataset,
    metrics=[faithfulness_metric, relevancy_metric],
    run_config=config
)

print("\n--- RESULTADOS DE LA EVALUACIÓN RAGAS ---")
print(result)

df = result.to_pandas()
df.to_csv("ragas_evaluation_results.csv", index=False)
print("\nResultados guardados exitosamente en 'ragas_evaluation_results.csv'.")