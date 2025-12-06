from src.functions.SystemFunctions import SystemFunctions
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.orchestrator_agent import OrchestratorAgent
from src.tools.SystemTools import build_functions_tools
from src.utils.pdf_to_text import pdf_to_text
from src.utils.data_extraction import base_documents

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
new_docs_path = "data/new_docs"

functions = SystemFunctions(docs_path, faiss_path)
tools = build_functions_tools(functions)

llm_orchestrator = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.0,
)
orchestrator = OrchestratorAgent(llm_orchestrator, tools)


async def start(update: Update, context):
    # Se inicializan las carpetas necesarias en caso de que no existan
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        base_documents(docs_path)

    if not os.path.exists(new_docs_path):
        os.makedirs(new_docs_path)

    if not os.path.exists(faiss_path) or not os.listdir(faiss_path):
        os.makedirs(faiss_path)
        functions.run_indexing()

    await update.message.reply_text("Hola, soy tu bot de deportes. Pregúntame lo que quieras =)")





# Handlers de texto
async def mensaje(update: Update, context):
    # Se recibe el mensaje del usuario
    texto = update.message.text
    print(f"Mensaje recibido: {texto}")

    # Se procesa la consulta con el orquestador
    respuesta = orchestrator.run(texto)
    #print("\n",respuesta)
    await update.message.reply_text(respuesta)



# Handlers de documentos (no implementado)
async def documento(update: Update, context):

    # Se recibe el documento
    doc = update.message.document
    file_name = doc.file_name
    print(f"Documento recibido: {file_name}")

    # Se descarga el archivo
    file = await doc.get_file()

    # Se guarda temporalmente el archivo en la carpeta new_docs
    file_path = os.path.join(new_docs_path, file_name)
    await file.download_to_drive(file_path)

    try: 
        # Se verifica que el documento sea un .txt o pdf(se transformará a txt)
        txt_path = pdf_to_text(file_name, file_path, docs_path)

        # Se reindexa el nuevo documento
        functions.run_indexing(new_doc_path=txt_path)


        # Se responde al usuario
        await update.message.reply_text(
            f"Archivo procesado y agregado al índice.\n"
            "Ahora puedes hacer preguntas sobre su contenido."
        )

    except ValueError as e:
        await update.message.reply_text(str(e))



def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, mensaje))
    app.add_handler(MessageHandler(filters.Document.ALL, documento))

    app.run_polling()   

if __name__ == "__main__":
    main()
