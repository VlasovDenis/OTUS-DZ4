from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


CHROMA_DIR = Path("data/chroma")

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3:4b"

QUERY = input("Введите вопрос: ").strip()

TOP_K = 2


def main():
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )

    vector_store = Chroma(
        collection_name="corporate_documents",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    documents = vector_store.similarity_search(
        QUERY,
        k=TOP_K,
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
{QUERY}

Дай короткий и точный ответ.
"""

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )

    response = llm.invoke(prompt)

    print(f"Вопрос: {QUERY}")
    print()
    print("Ответ:")
    print(response.content)
    print()

    print("Источники:")
    for document in documents:
        print(f"- {document.metadata['source']}")


if __name__ == "__main__":
    main()