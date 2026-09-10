from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


CHROMA_DIR = Path("data/chroma")
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

QUERY = "Сколько дней хранятся резервные копии?"


def main():
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )

    vector_store = Chroma(
        collection_name="corporate_documents",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    results = vector_store.similarity_search_with_score(
        QUERY,
        k=3,
    )

    print(f"Вопрос: {QUERY}")
    print()

    for index, (document, score) in enumerate(results, start=1):
        print(f"===== RESULT {index} =====")
        print(f"Источник: {document.metadata['source']}")
        print(f"Distance: {score:.4f}")
        print()
        print(document.page_content)
        print()


if __name__ == "__main__":
    main()