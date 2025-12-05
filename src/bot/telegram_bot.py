from src.functions.SystemFunctions import SystemFunctions
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.orchestrator_agent import OrchestratorAgent
from src.tools.SystemTools import build_functions_tools

from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters
from telegram import Update
from dotenv import load_dotenv
import os

# Variables de entorno para el bot de Telegram
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Variables para el orquestador
docs_path = "data/raw_docs"
faiss_path = "data/faiss_index"

functions = SystemFunctions(docs_path, faiss_path)
tools = build_functions_tools(functions)

llm_orchestrator = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.0,
)
orchestrator = OrchestratorAgent(llm_orchestrator, tools)


async def start(update: Update, context):
    await update.message.reply_text("Hola, soy tu bot de deportes. Pregúntame lo que quieras =)")

async def mensaje(update: Update, context):
    # Se recibe el mensaje del usuario
    texto = update.message.text
    print(f"Mensaje recibido: {texto}")

    # Se procesa la consulta con el orquestador
    respuesta = orchestrator.run(texto)
    #print("\n",respuesta)
    await update.message.reply_text(respuesta)

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, mensaje))

    app.run_polling()   

if __name__ == "__main__":
    main()
