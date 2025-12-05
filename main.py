from src.functions.SystemFunctions import SystemFunctions
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.orchestrator_agent import OrchestratorAgent
from src.tools.SystemTools import build_functions_tools
from src.bot.telegram_bot import main as run_telegram_bot
import os

# Cargamos las variables de entorno
from dotenv import load_dotenv
load_dotenv()


## Eliminar##
from pathlib import Path

# def print_docs_pretty(docs):
#     for i, doc in enumerate(docs, start=1):
#         source = doc.metadata.get("source", "desconocido")
#         filename = Path(source).name          # Fútbol.txt
#         title = Path(source).stem             # Fútbol (sin .txt)
#         snippet = doc.page_content[:150].replace("\n", " ")

#         print(f"\n--- Documento {i} ---")
#         print(f"Título: {title}")
#         print(f"Archivo: {filename}")
#         print(f"Snippet: {snippet}...")

if __name__ == "__main__":
    docs_path = "data/raw_docs"
    faiss_path = "data/faiss_index"


    # ## AQUI SOLO DEBERIA IR EL ORQUESTADOR Y EL BOT ##
    # # Inicializamos el bot
    run_telegram_bot()
    
    # # Creamos el functions principal del sistema - Esto se quita luego
    # functions = SystemFunctions(docs_path, faiss_path)

    # # Ejemplo de consulta para clasificación de intención- Esto se quita luego
    # pregunta = "Como se compara el ajedrez con el futbol?"
    # functions.run_classification(pregunta)

    # # Ejemplo de consulta para recuperación de documentos - Esto se quita luego
    # docs = functions.run_retrieval(pregunta, use_llm=False)
    # print_docs_pretty(docs)

    # # Ejemplo de consulta para RAG - Esto se quita luego
    # response = functions.run_rag_respose(pregunta, use_llm=False)

    # print("\n--- Contexto RAG ---")
    # print(response["context"])
    # print("\n--- Respuesta RAG ---")
    # print(response["answer"])
    
    # # Ejemplo de evaluación de respuesta - Esto se quita luego
    # evaluation = functions.run_evaluation(
    #     pregunta,
    #     response["context"],
    #     response["answer"],
    # )
    # print("\n--- Evaluación de la respuesta ---")
    # print("Veredicto: ",evaluation["veredicto"])
    # print("\nExplicacion: ",evaluation["explicacion"])
    # LLM para el orquestador (puede ser el mismo modelo que ya usas)

    # llm_orchestrator = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     google_api_key=os.environ["GOOGLE_API_KEY"],
    #     temperature=0.0,
    # )
    # functions = SystemFunctions(docs_path, faiss_path)

    # # Lista de tools disponibles
    # tools = build_functions_tools(functions)

    # # Crear el agente orquestador
    # orchestrator = OrchestratorAgent(llm_orchestrator, tools)

    # # Ejemplo de uso
    # pregunta = "Hola, como estas?"
    # respuesta = orchestrator.run(pregunta)

    # print("\nRespuesta final del orquestador:\n")
    # print(respuesta)
