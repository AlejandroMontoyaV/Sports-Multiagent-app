import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


class IndexerAgent:
    """
    Agente de consumo e indexación.
    Se encarga de:
        - Cargar documentos TXT
        - Limpiar y segmentar en chunks
        - Crear embeddings
        - Construir y guardar el índice FAISS
    """

    # Constructor
    def __init__(self, docs_path: str, faiss_path: str, embeddings):
        """
        Parámetros:
        - raw_docs_path: Carpeta donde están los .txt
        - faiss_output_path: Carpeta donde se guardará el índice FAISS
        """
        self.docs_path = docs_path
        self.faiss_path = faiss_path
        self.embeddings = embeddings

    # Cargar documentos
    def load_documents(self):
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.txt",
            loader_cls=lambda path: TextLoader(path, encoding="utf-8")
        )
        documents = loader.load()
        return documents

    # Dividir documentos en chunks de 1000 tokens
    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        return texts

    # Crear y guardar el índice FAISS
    def create_index(self, texts):
        vector_store = FAISS.from_documents(texts, self.embeddings)
        vector_store.save_local(self.faiss_path)

    # Ejecutar el agente
    def run(self):
        documents = self.load_documents()
        texts = self.split_documents(documents)
        self.create_index(texts)