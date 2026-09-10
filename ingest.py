from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DOCUMENTS_DIR = Path("data/documents")
CHROMA_DIR = Path("data/chroma")

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents():
    documents = []

    for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8-sig")

        document = Document(
            page_content=text,
            metadata={
                "source": file_path.name,
            },
        )

        documents.append(document)

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return text_splitter.split_documents(documents)


def create_vector_database(chunks):
    print(f"Загрузка embedding-модели: {EMBEDDING_MODEL}")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )
    
    # Для учебного проекта при повторном запуске
    # пересоздаем векторную БД с нуля.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="corporate_documents",
        collection_metadata={
            "hnsw:space": "cosine",
        },
    )

    return vector_store


def main():
    documents = load_documents()
    print(f"Загружено документов: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Создано чанков: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(
            f"Chunk {index}: "
            f"{chunk.metadata['source']} "
            f"({len(chunk.page_content)} символов)"
        )

    vector_store = create_vector_database(chunks)

    print()
    print("Векторная база создана.")
    print(f"Количество записей: {vector_store._collection.count()}")
    print(f"Каталог: {CHROMA_DIR}")


if __name__ == "__main__":
    main()