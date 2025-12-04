from src.agents.indexer_agent import IndexerAgent
from src.agents.classifier_agent import ClassifierAgent
from langchain_google_genai import ChatGoogleGenerativeAI
import os

class SystemPipeline:
    # Constructor
    def __init__(self, docs_path: str, faiss_path: str):
        self.indexer_agent = IndexerAgent(docs_path, faiss_path)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        self.classifier_agent = ClassifierAgent(llm)

    # Ejecutar la indexación
    def run_indexing(self):
        print("Iniciando el proceso de indexación...")
        self.indexer_agent.run()
        print("Indexación completada.")

    # Ejecutar la clasificación de intención
    def run_classification(self, query: str):
        intent = self.classifier_agent.classify_intent(query)
        print(f"Intención clasificada: {intent}")