"""Fase 0: mide el pipeline ORIGINAL del Avance 2 con Groq (índice viejo 'chroma_db', MiniLM inglés, k=4, prompt original).
Guarda respuestas y contextos REALES + llamadas al LLM, tokens y tiempo.

Uso:  python medir_base.py [salida.json] [pausa_segundos]
"""
import json
import sys
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import get_llm  # también carga .env


class Contador(BaseCallbackHandler):
    """Cuenta llamadas al LLM y tokens."""
    def __init__(self):
        self.llamadas = 0
        self.tokens = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.llamadas += 1

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                um = getattr(getattr(g, "message", None), "usage_metadata", None)
                if um:
                    self.tokens += um.get("total_tokens", 0)


def cadena_avance2():
    # Réplica fiel del Avance 2: NO cambiar nada aquí, es la línea base.
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma(persist_directory="chroma_db", embedding_function=emb)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(temperature=0.2)
    system_prompt = (
        "Eres Ragi, un asistente académico experto. Responde con rigor científico "
        "y basándote estrictamente en el contexto proporcionado. Si la información "
        "no está en los documentos, indícalo claramente y no inventes datos.\n\n"
        "Contexto:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))


def main():
    salida = sys.argv[1] if len(sys.argv) > 1 else "baseline_avance2.json"
    pausa = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    data = json.load(open("test_dataset.json", encoding="utf-8"))
    chain = cadena_avance2()

    resultados = []
    for i, item in enumerate(data, 1):
        q = item["user_input"]
        print(f"[{i}/{len(data)}] {q[:70]}")
        c = Contador()
        t0 = time.time()
        try:
            r = chain.invoke({"input": q, "chat_history": []}, config={"callbacks": [c]})
            resp = r["answer"]
            ctx = [d.page_content for d in r["context"]]
            fuentes = [f"{d.metadata.get('source','?')} pág.{d.metadata.get('page','?')}" for d in r["context"]]
            error = None
        except Exception as e:  # seguimos aunque una pregunta falle
            resp, ctx, fuentes, error = "", [], [], str(e)
            print("   ERROR:", e)
        resultados.append({
            "user_input": q, "tipo": item.get("tipo", "base"), "reference": item["reference"],
            "response": resp, "retrieved_contexts": ctx, "fuentes": fuentes,
            "llamadas_llm": c.llamadas, "tokens": c.tokens,
            "tiempo_s": round(time.time() - t0, 2), "error": error,
        })
        time.sleep(pausa)

    ok = [r for r in resultados if not r["error"]]
    n = max(len(ok), 1)
    resumen = {
        "preguntas": len(resultados), "con_error": len(resultados) - len(ok),
        "llamadas_llm_promedio": round(sum(r["llamadas_llm"] for r in ok) / n, 2),
        "tokens_promedio": round(sum(r["tokens"] for r in ok) / n, 1),
        "tiempo_promedio_s": round(sum(r["tiempo_s"] for r in ok) / n, 2),
    }
    json.dump({"resumen": resumen, "resultados": resultados}, open(salida, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nRESUMEN:", resumen, f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
