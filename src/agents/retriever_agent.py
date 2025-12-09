from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
import os



class RetrieverAgent:
    """
    Agente de recuperación de información.
    Se encarga de:
        - Cargar el índice FAISS
        - Recuperar documentos relevantes para una consulta
    """

    # Constructor
    def __init__(self, faiss_path: str, llm: BaseChatModel,embeddings, k: int =5):
        """
        Parámetros:
        - faiss_path: Carpeta donde está guardado el índice FAISS
        - k: Número de documentos a recuperar(por defecto 5)
        """
        self.faiss_path = faiss_path
        self.k = k
        self.embeddings = embeddings
        self.llm = llm

    # Cargar el índice FAISS al iniciar el agente
    def _load_vector_store(self):
        """
        Carga el índice FAISS desde disco.
        Se llama en cada búsqueda para que siempre use la versión actualizada.
        """
        index_file = os.path.join(self.faiss_path, "index.faiss")
        store_file = os.path.join(self.faiss_path, "index.pkl")

        if os.path.exists(index_file) and os.path.exists(store_file):
            return FAISS.load_local(
                self.faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            return None
    # Función que usa un LLM para mejorar la velocidad de recuperación
    def rewrite_query(self, query: str) -> str:
        """
        Usa el LLM para reescribir la consulta de usuario
        en una forma más corta y adecuada para búsqueda semántica.
        """
        prompt = f"""
        Reescribe la siguiente consulta del usuario de forma breve, clara
        y adecuada para búsqueda semántica en documentos sobre clima
        y cambio climático. No cambies el significado.

        Consulta original: \"\"\"{query}\"\"\"

        Nueva consulta:
        """
        response = self.llm.invoke(prompt)
        new_query = response.content.strip()
        return new_query or query

    # Recuperar documentos relevantes
    def retrieve_documents(self, query: str, use_llm: bool = True):
        #Cargar SIEMPRE el índice desde disco
        vector_store = self._load_vector_store()
        if vector_store is None:
            raise RuntimeError(
                "El índice FAISS no está disponible. Asegúrate de haber indexado documentos."
            )

        # Si se indica, reescribir la consulta con el LLM
        search_query = self.rewrite_query(query) if use_llm else query

        docs = vector_store.similarity_search(search_query, self.k)
        return docs
