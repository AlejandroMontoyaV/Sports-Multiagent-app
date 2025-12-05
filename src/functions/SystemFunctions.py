from src.agents.indexer_agent import IndexerAgent
from src.agents.classifier_agent import ClassifierAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.rag_agent import RagAgent
from src.agents.evaluator_agent import EvaluatorAgent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os


# AQUI SE INICIALIZAN LOS AGENTE
#ESTA TODo CENTRALIZADO
class SystemFunctions:
    # Constructor
    def __init__(self, docs_path: str, faiss_path: str):
        
        # Inicializar embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        # Inicializar LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0.0,
        )
        # Inicializar agentes
        self.indexer_agent = IndexerAgent(docs_path, faiss_path, self.embeddings)
        self.classifier_agent = ClassifierAgent(llm)
        self.retriever_agent = RetrieverAgent(faiss_path, llm, self.embeddings, k=10)
        self.rag_agent = RagAgent(self.retriever_agent, llm)
        self.evaluator_agent = EvaluatorAgent(llm)

    # Ejecutar la indexación
    def run_indexing(self):
        print("Iniciando el proceso de indexación...")
        self.indexer_agent.run()
        print("Indexación completada.")

    # Ejecutar la clasificación de intención
    def run_classification(self, query: str):
        intent = self.classifier_agent.classify_intent(query)
        #print(f"Intención clasificada: {intent}")
        return intent

    # Ejecutar la recuperación de documentos
    def run_retrieval(self, query: str, use_llm: bool = False):
        documents = self.retriever_agent.retrieve_documents(query, use_llm)
        return documents
    
    # Ejecutar la generación de respuesta RAG
    def run_rag_respose(self, query: str, use_llm: bool = False):
        result = self.rag_agent.generate_response(query)
        return result
    
    def run_evaluation(self, query: str, context: str, answer: str):
        evaluation = self.evaluator_agent.evaluate_response(query, context, answer)
        return evaluation

    