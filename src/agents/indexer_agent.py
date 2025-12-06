import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.utils.data_extraction import base_documents


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
    def load_documents(self, new_doc_path: str | None = None):
        # Si solo se quiere cargar un documento nuevo
        if new_doc_path is not None:
                    # Cargar un único archivo
                    loader = TextLoader(new_doc_path, encoding="utf-8")
                    return loader.load()

        # Si hay que cargar todos los documentos
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
    def run(self, new_doc_path: str | None = None):
        """
        Si el índice NO existe aún:
            - Genera los documentos base
            - Indexa TODOS los .txt de docs_path

        Si el índice SÍ existe:
            - Si new_doc_path es None -> (opcional) podrías reindexar todo,
              pero aquí solo voy a añadir TODO lo de docs_path de nuevo si lo deseas.
            - Si new_doc_path tiene ruta -> añade SOLO ese archivo al índice.
        """
        
        index_file = os.path.join(self.faiss_path, "index.faiss")
        store_file = os.path.join(self.faiss_path, "index.pkl")

        # ¿La carpeta del índice está vacía?
        index_vacio = not (os.path.exists(index_file) and os.path.exists(store_file))

        # Si el indice no existe, crearlo -> No debería pasar, se crea al inicio del bot
        if index_vacio:
            # Primera vez: crear índice completo
            print("Creando índice FAISS por primera vez...")
            # Cargar y procesar documentos (todos los .txt de docs_path)
            documents = self.load_documents()
            texts = self.split_documents(documents)
            self.create_index(texts)
            print("Índice creado por primera vez.")
            return

        # Si ya existe el índice:
        # Cargar el índice existente
        vector_store = FAISS.load_local(
            self.faiss_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        if new_doc_path is None:
            # MODO: añadir TODO lo que haya en docs_path (ojo con duplicados)
            print("Añadiendo documentos de la carpeta completa al índice (puede duplicar).")
            documents = self.load_documents()          # todos los .txt
        else:
            # MODO: añadir SOLO el nuevo archivo
            print(f"Añadiendo solo el documento: {new_doc_path}")
            documents = self.load_documents(new_doc_path)

        texts = self.split_documents(documents)
        vector_store.add_documents(texts)
        vector_store.save_local(self.faiss_path)
        print("Índice FAISS actualizado.")
        return
        