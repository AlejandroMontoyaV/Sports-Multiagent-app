from src.pipelines.SystemPipeline import SystemPipeline
from src.bot.telegram_bot import main as run_telegram_bot

# Cargamos las variables de entorno
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    docs_path = "data/raw_docs"
    faiss_path = "data/faiss_index"

    ## AQUI SOLO DEBERIA IR EL ORQUESTADOR Y EL BOT ##
    # Inicializamos el bot
    #run_telegram_bot()
    
    # Creamos el pipeline principal del sistema - Esto se quita luego
    pipeline = SystemPipeline(docs_path, faiss_path)

    # Ejemplo de consulta para clasificación de intención- Esto se quita luego
    pregunta = "Que dice el documento de ajedrez?"
    pipeline.run_classification(pregunta)