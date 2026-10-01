from pathlib import Path
from typing import List, Tuple

from langchain.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



_vector_store = None


def load_policy_documents():
    policies_dir = Path(__file__).resolve().parents[2] / "policies"
    documents = []
    for txt_file in policies_dir.glob("*.txt"):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = txt_file.name
        documents.extend(docs)
    return documents


def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        add_start_index=True,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def get_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    persist_dir = Path(__file__).resolve().parent.parent.parent / "chroma_db"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if persist_dir.exists() and any(persist_dir.iterdir()):
        _vector_store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
        print(f"Loaded existing vector store from {persist_dir}")
    else:
        print("Loading policy documents...")
        documents = load_policy_documents()
        if not documents:
            policies_dir = Path(__file__).resolve().parent.parent.parent / "policies"
            raise ValueError(f"No policy documents found in {policies_dir}")
        print(f"   Found {len(documents)} documents")
        print("Splitting documents into chunks...")
        chunks = split_documents(documents)
        print(f"   Created {len(chunks)} chunks")
        print("Creating embeddings and storing in ChromaDB...")
        _vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir),
        )
        print(f"Created vector store with {len(chunks)} chunks")

    return _vector_store

@tool(response_format="content_and_artifact")
def search_policies(query: str)  -> Tuple[str, List[Document]]:
    """
    Search the store's policy documents for information.

    Use this tool when you need to find information about:
    - Returns policy (return window, eligibility, process)
    - Refunds (timeline, methods, amounts)
    - Shipping policy (delivery times, fees, tracking)
    - Order cancellations (before/after shipping)
    - General FAQ about returns and cancellations

    Args:
        query: The question or topic to search for in policy documents

    Returns:
        Relevant information from the policy documents

    """

    vector_store = get_vector_store()
    retrieved_docs = vector_store.similarity_search(query, k=4)
    if not retrieved_docs:
        return "No relevant policy information found.", []
    serialized = "\n\n---\n\n".join([
        f"**Source: {doc.metadata.get('source', 'Unknown')}**\n{doc.page_content}"
        for doc in retrieved_docs
    ])
    return serialized, retrieved_docs

def initialize_vector_store():
    get_vector_store()


__all__ = ["search_policies", "get_vector_store", "initialize_vector_store"]