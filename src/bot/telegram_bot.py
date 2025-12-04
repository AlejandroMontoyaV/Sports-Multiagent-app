import os
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters
from telegram import Update
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

async def start(update: Update, context):
    await update.message.reply_text("Hola, soy tu bot. Envíame un mensaje.")

async def mensaje(update: Update, context):
    texto = update.message.text
    await update.message.reply_text(f"Recibí: {texto}")
    print(f"Mensaje recibido: {texto}")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT, mensaje))

    app.run_polling()   

if __name__ == "__main__":
    main()
