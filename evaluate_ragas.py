"""Evalúa con Ragas (4 métricas) un JSON producido por medir_base.py (o, luego, por el agente).
Excluye las preguntas 'fuera_de_alcance' (no tienen sentido para Ragas; son casos de comportamiento).

Uso:  python evaluate_ragas.py baseline_avance2.json ragas_avance2.csv
"""
import sys
import types

# --- Parche VertexAI para ragas (como en tu versión anterior) ---
for mod_name in ["langchain_community.chat_models.vertexai", "langchain_community.chat_models"]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        if "vertexai" in mod_name:
            m.ChatVertexAI = type("ChatVertexAI", (object,), {})
        sys.modules[mod_name] = m

import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy

try:
    from ragas.metrics import LLMContextPrecisionWithReference as ContextPrecision, LLMContextRecall as ContextRecall
except ImportError:
    from ragas.metrics import ContextPrecision, ContextRecall

from config import LLM_MODEL, get_embeddings, get_llm


def main():
    entrada = sys.argv[1] if len(sys.argv) > 1 else "baseline_avance2.json"
    salida = sys.argv[2] if len(sys.argv) > 2 else "ragas_resultados.csv"
    desde = int(sys.argv[3]) if len(sys.argv) > 3 else 0          # para evaluar por tandas: python evaluate_ragas.py a.json s.csv 0 5
    hasta = int(sys.argv[4]) if len(sys.argv) > 4 else None
    max_ctx = int(os.getenv("MAX_CTX", "6"))                       # tope de fragmentos por pregunta (controla tokens del juez)

    raw = json.load(open(entrada, encoding="utf-8"))
    filas = raw["resultados"] if isinstance(raw, dict) else raw
    filas = [r for r in filas if r.get("tipo", "base") == "base" and not r.get("error") and r["retrieved_contexts"]]
    for r in filas:   # el agente puede repetir el mismo fragmento en varias búsquedas: se quitan duplicados y se limita
        r["retrieved_contexts"] = list(dict.fromkeys(r["retrieved_contexts"]))[:max_ctx]
    filas = filas[desde:hasta]
    print(f"Evaluando {len(filas)} preguntas (filas {desde} a {hasta or 'fin'}) con 4 métricas...")

    ds = Dataset.from_dict({
        "user_input": [r["user_input"] for r in filas],
        "response": [r["response"] for r in filas],
        "retrieved_contexts": [r["retrieved_contexts"] for r in filas],
        "reference": [r["reference"] for r in filas],
    })

    modelo_juez = os.getenv("EVAL_MODEL", LLM_MODEL)   # puedes probar otro más rápido: set EVAL_MODEL=...
    print(f"Modelo juez: {modelo_juez}")
    llm = LangchainLLMWrapper(get_llm(modelo_juez))
    emb = LangchainEmbeddingsWrapper(get_embeddings())   # local: no gasta cuota para answer_relevancy

    result = evaluate(
        ds,
        metrics=[Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=emb, strictness=1),
                 ContextPrecision(llm=llm), ContextRecall(llm=llm)],
        run_config=RunConfig(timeout=180, max_retries=6, max_workers=1),
    )
    df = result.to_pandas()
    if os.path.exists(salida):   # acumula tandas; si una pregunta se repite, gana la más reciente
        import pandas as pd
        df = pd.concat([pd.read_csv(salida), df]).drop_duplicates(subset=["user_input"], keep="last")
    df.to_csv(salida, index=False)
    nums = df.select_dtypes("number")
    print("\n--- PROMEDIOS (acumulado en el CSV) ---")
    print(nums.mean().round(3))
    print("\nValores vacíos por métrica (si hay, repite esas filas):")
    print(nums.isna().sum())
    print(f"\nPreguntas en el CSV: {len(df)}\nGuardado en {salida}")


if __name__ == "__main__":
    main()
