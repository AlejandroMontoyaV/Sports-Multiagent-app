from src.pipelines.SystemPipeline import SystemPipeline
# Cargamos las variables de entorno
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    docs_path = "data/raw_docs"
    faiss_path = "data/faiss_index"
    
    # Creamos el pipeline principal del sistema
    pipeline = SystemPipeline(docs_path, faiss_path)

    # 1. Ejecutamos la indexación
    pipeline.run_indexing()

    # 2. Clasificamos la intención (pendiente de implementación)