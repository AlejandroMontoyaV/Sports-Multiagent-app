# Ignore warnings about unsupported 'title' key in schema
import logging
logging.basicConfig(level=logging.ERROR)

from src.bot.telegram_bot import main as run_telegram_bot

# Cargamos las variables de entorno
from dotenv import load_dotenv
load_dotenv()

class IgnoreTitleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Key 'title' is not supported in schema" not in record.getMessage()


if __name__ == "__main__":
    # Inicializamos las rutas y el sistema de funciones
    logger = logging.getLogger("google.genai")
    logger.addFilter(IgnoreTitleFilter())

    print("Bot de Telegram iniciado.")
    run_telegram_bot()
    print("Bot de Telegram terminado.")
    
 
