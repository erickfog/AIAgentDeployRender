import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

from src.agent import build_agent_executor

load_dotenv()


st.set_page_config(page_title="Agente de apostas", page_icon="📊")
st.title("📊 Agente de apostas (odds e calculos)")
st.caption(
    "Assistente educativo: odds decimais, retorno, parlay e jogo responsavel — LangChain no Render."
)


@st.cache_resource
def get_executor():
    return build_agent_executor()


def ensure_env() -> bool:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Defina a variavel OPENAI_API_KEY para usar o agente.")
        return False
    return True


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if ensure_env():
        executor = get_executor()
        history = []
        for message in st.session_state.messages[:-1]:
            if message["role"] == "user":
                history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                history.append(AIMessage(content=message["content"]))

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                result = executor.invoke({"input": user_input, "chat_history": history})
                answer = result.get("output", "Nao consegui gerar resposta agora.")
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
