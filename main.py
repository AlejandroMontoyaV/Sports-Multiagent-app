from src.pipelines.SystemPipeline import SystemPipeline
from src.bot.telegram_bot import main as run_telegram_bot

# Cargamos las variables de entorno
from dotenv import load_dotenv
load_dotenv()


## Eliminar##
from pathlib import Path

def print_docs_pretty(docs):
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "desconocido")
        filename = Path(source).name          # Fútbol.txt
        title = Path(source).stem             # Fútbol (sin .txt)
        snippet = doc.page_content[:150].replace("\n", " ")

        print(f"\n--- Documento {i} ---")
        print(f"Título: {title}")
        print(f"Archivo: {filename}")
        print(f"Snippet: {snippet}...")

if __name__ == "__main__":
    docs_path = "data/raw_docs"
    faiss_path = "data/faiss_index"

    ## AQUI SOLO DEBERIA IR EL ORQUESTADOR Y EL BOT ##
    # Inicializamos el bot
    #run_telegram_bot()
    
    # Creamos el pipeline principal del sistema - Esto se quita luego
    pipeline = SystemPipeline(docs_path, faiss_path)

    # Ejemplo de consulta para clasificación de intención- Esto se quita luego
    pregunta = "Como se compara el ajedrez con el futbol?"
    pipeline.run_classification(pregunta)

    # Ejemplo de consulta para recuperación de documentos - Esto se quita luego
    docs = pipeline.run_retrieval(pregunta, use_llm=False)
    print_docs_pretty(docs)

    # Ejemplo de consulta para RAG - Esto se quita luego
    response = pipeline.run_rag_respose(pregunta, use_llm=False)
    print("\n--- Respuesta RAG ---")
    print(response)