from src.agents.indexer_agent import IndexerAgent

class SystemPipeline:
    # Constructor
    def __init__(self, docs_path: str, faiss_path: str):
        self.indexer_agent = IndexerAgent(docs_path, faiss_path)

    # Ejecutar la indexación
    def run_indexing(self):
        print("Iniciando el proceso de indexación...")
        self.indexer_agent.run()
        print("Indexación completada.")