import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag_chain import get_conversational_rag_chain

st.set_page_config(page_title="Ragi - Asistente Académico", page_icon="🎓")
st.title("🎓 Ragi: Asistente Académico RAG")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = get_conversational_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

user_query = st.chat_input("Escribe tu pregunta sobre los documentos académicos...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analizando documentos..."):
            response = st.session_state.rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            answer = response["answer"]
            st.markdown(answer)

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=answer))