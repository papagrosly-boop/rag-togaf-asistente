"""Ejecuta el agente sobre test_dataset.json y guarda un JSON con el MISMO formato que medir_base.py
(así evaluate_ragas.py sirve para ambos). retrieved_contexts = fragmentos que devolvieron las tools.

Uso:  python medir_agente.py [salida.json] [pausa_segundos]
"""
import json
import sys
import time

from agent import Agente


def main():
    salida = sys.argv[1] if len(sys.argv) > 1 else "agente_resultados.json"
    pausa = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0   # respeta el tope de tokens por minuto
    data = json.load(open("test_dataset.json", encoding="utf-8"))
    ag = Agente(log_path="logs/uso_tools_eval.jsonl")

    resultados = []
    for i, item in enumerate(data, 1):
        q = item["user_input"]
        print(f"[{i}/{len(data)}] {q[:70]}")
        try:
            r = ag.responder(q)
            error = None
        except Exception as e:
            r, error = {"respuesta": "", "contextos": [], "fuentes_base": [], "fuentes_web": [], "tools": [],
                        "llamadas_llm": 0, "tokens": 0, "tiempo_s": 0}, str(e)
            print("   ERROR:", e)
        resultados.append({
            "user_input": q, "tipo": item.get("tipo", "base"), "reference": item["reference"],
            "response": r["respuesta"], "retrieved_contexts": r["contextos"],
            "fuentes": r["fuentes_base"] + [w["url"] for w in r["fuentes_web"]],
            "tools_usadas": [{"nombre": t["nombre"], "args": t["args"]} for t in r["tools"]],
            "llamadas_llm": r["llamadas_llm"], "tokens": r["tokens"], "tiempo_s": r["tiempo_s"], "error": error,
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
    json.dump({"resumen": resumen, "resultados": resultados}, open(salida, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nRESUMEN:", resumen, f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
