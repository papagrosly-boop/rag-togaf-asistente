import json
import streamlit as st
from agent import Agente, historial_a_mensajes

st.set_page_config(page_title="Ragi - Agente TOGAF", page_icon="🎓")
st.title("🎓 Ragi: Agente sobre TOGAF 9.1")
st.caption("Busca primero en la base de documentos; si no está y es información pública, consulta sitios oficiales de The Open Group.")


@st.cache_resource(show_spinner="Cargando agente...")
def cargar_agente():
    return Agente()


agente = cargar_agente()
if "chat" not in st.session_state:
    st.session_state.chat = []   # [{"rol", "texto", "meta"}]


def mostrar_meta(meta):
    with st.expander("🔧 Tools usadas"):
        if not meta["tools"]:
            st.markdown("_Ninguna (respuesta sin herramientas)_")
        for t in meta["tools"]:
            st.markdown(f"**{t['nombre']}**")
            st.code(json.dumps(t["args"], ensure_ascii=False), language="json")
        st.caption(f"{meta['llamadas_llm']} llamadas al LLM · {meta['tokens']} tokens · {meta['tiempo_s']} s")
    with st.expander("📚 Fuentes"):
        st.markdown("**Base de conocimientos**")
        for f in meta["fuentes_base"] or ["_(ninguna citada)_"]:
            st.markdown(f"- {f}")
        st.markdown("**Web**")
        for w in meta["fuentes_web"] or []:
            st.markdown(f"- [{w['titulo']}]({w['url']})")
        if not meta["fuentes_web"]:
            st.markdown("- _(ninguna)_")


for m in st.session_state.chat:
    with st.chat_message(m["rol"]):
        st.markdown(m["texto"])
        if m.get("meta"):
            mostrar_meta(m["meta"])

pregunta = st.chat_input("Pregunta sobre TOGAF 9.1...")
if pregunta:
    with st.chat_message("user"):
        st.markdown(pregunta)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Pensando..."):
                hist = historial_a_mensajes([(m["rol"], m["texto"]) for m in st.session_state.chat])
                r = agente.responder(pregunta, hist)
            st.markdown(r["respuesta"])
            mostrar_meta(r)
            meta = {k: r[k] for k in ("tools", "fuentes_base", "fuentes_web", "llamadas_llm", "tokens", "tiempo_s")}
        except Exception as e:
            msg = ("Se alcanzó el límite de uso del modelo. Espera unos minutos y vuelve a intentar."
                   if "429" in str(e) or "rate" in str(e).lower()
                   else "Ocurrió un error al procesar la pregunta. Intenta de nuevo.")
            st.error(msg)
            st.caption(f"Detalle técnico: {str(e)[:300]}")
            r, meta = {"respuesta": msg}, None
    st.session_state.chat.append({"rol": "user", "texto": pregunta, "meta": None})
    st.session_state.chat.append({"rol": "assistant", "texto": r["respuesta"], "meta": meta})
