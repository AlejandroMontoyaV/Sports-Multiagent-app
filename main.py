from src.bot.telegram_bot import main as run_telegram_bot
from src.utils.data_extraction import base_documents
from src.functions.SystemFunctions import SystemFunctions
import os

# Cargamos las variables de entorno
from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    # Inicializamos las rutas y el sistema de funciones
    
    run_telegram_bot()
    
 
