from pathlib import Path
import time

import streamlit as st

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


CHROMA_DIR = Path("data/chroma")

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3:4b"


embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
)

vector_store = Chroma(
    collection_name="corporate_documents",
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
)

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
)


st.set_page_config(
    page_title="Enterprise Private GPT",
    page_icon="🔒",
)

st.title("🔒 Enterprise Private GPT")

st.write(
    "Локальный корпоративный RAG-ассистент "
    "на базе Ollama, Qwen3 и ChromaDB."
)

top_k = st.slider(
    "Количество документов Top-K",
    min_value=1,
    max_value=3,
    value=2,
)

question = st.text_input(
    "Введите вопрос по корпоративной базе знаний:"
)


if st.button("Отправить") and question:
    start_time = time.time()

    documents = vector_store.similarity_search(
        question,
        k=top_k,
    )

    context = "\n\n---\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
Ты корпоративный AI-ассистент.

Отвечай на вопрос только на основании предоставленного контекста.
Не используй собственные знания для дополнения ответа.

Если в контексте нет информации для ответа, скажи:
"В корпоративной базе знаний нет информации для ответа на этот вопрос."

КОНТЕКСТ:
{context}

ВОПРОС:
{question}

Дай короткий и точный ответ.
"""

    response = llm.invoke(prompt)

    elapsed_time = time.time() - start_time

    st.subheader("Ответ")
    st.write(response.content)

    st.subheader("Источники")

    for document in documents:
        st.write(f"- {document.metadata['source']}")

    st.caption(
        f"Время ответа: {elapsed_time:.2f} сек."
    )