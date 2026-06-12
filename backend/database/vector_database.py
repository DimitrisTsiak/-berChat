from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import uuid


def create_vector_database(file_path:str, db_name:str):
    loader  = PyPDFLoader(file_path=file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    )
    splitted_docs = splitter.split_documents(docs)
    
    text_chunks = [doc.page_content for doc in splitted_docs]
    client = chromadb.Client()
    client = chromadb.PersistentClient(path="./chroma_data")

    collection = client.get_or_create_collection(name=db_name)

    collection.add(
        ids=[str(uuid.uuid4()) for _ in text_chunks],
        documents=text_chunks
    )

    print("Vector Database created")

    return "ok"

def retrieve_vectors(query: str, db_name: str, n_results=5):
    client = chromadb.PersistentClient(path="./chroma_data")
    collection = client.get_collection(name=db_name)
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results
