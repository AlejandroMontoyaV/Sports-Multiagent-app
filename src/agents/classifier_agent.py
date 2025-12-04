from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict
import json
import re


CATEGORIES_DESCRIPTION = """
- "busqueda": el usuario quiere encontrar información específica
  (hechos, definiciones, datos concretos).
- "resumen": el usuario quiere que se resuma uno o varios textos/temas.
- "comparacion": el usuario quiere comparar dos o más cosas
  (por ejemplo, "compara A y B", "diferencias entre A y B").
- "general": pregunta abierta o explicativa que no encaja claro
  en las anteriores.
""".strip()


class ClassifierAgent:
    """
    Agente de clasificación de intención.
    Se encarga de:
        - Recibir una consulta del usuario
        - Clasificar la intención usando un modelo de lenguaje
    """

    # Constructor
    def __init__(self, llm: BaseChatModel):
        """
        Parámetros:
        - llm: Modelo de lenguaje para clasificación
        """
        self.llm = llm

    def build_messages(self, query: str):
        # Definir el prompt del sistema
        system_prompt = f"""
            Eres un clasificador de intención de consultas para un asistente
            sobre clima y cambio climático.

            Tu tarea es leer la consulta del usuario y clasificarla EXACTAMENTE
            en UNA de estas categorías:

            {CATEGORIES_DESCRIPTION}

            Instrucciones:
            - Devuelve SOLO un objeto JSON válido.
            - El JSON debe tener exactamente estas claves:
                - "category": uno de "busqueda", "resumen", "comparacion", "general"
                - "reason": breve explicación (1-3 frases) de tu decisión
            - No escribas texto fuera del JSON.
            """.strip()

        # Definir el prompt del usuario
        user_query = f"Consulta del usuario: {query}\nCategoría:"

        # Construir los mensajes
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
        return messages
    
    
    
    
    # Clasificar intención
    def classify_intent(self, query: str) -> Dict[str, str]:
        # Se definen los mensajes y la respuesta del LLM
        messages = self.build_messages(query)
        response = self.llm.invoke(messages)
        raw_content = response.content


        # Función para extraer JSON del texto(el llm lo daba rarito)
        def extract_json(text):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return match.group(0) if match else None

        # Se intenta extraer y parsear el JSON
        raw_json = extract_json(raw_content)
        if not raw_json:
            return {
                "category": "Error",
                "reason": f"El modelo no devolvió JSON: {raw_content}"
            }

        try:
            intent_data = json.loads(raw_json)
        except Exception as e:
            return {
                "category": "Error",
                "reason": f"Error analizando JSON: {e}"
            }
        
        # Se limpian y retornan los datos
        category = str(intent_data.get("category", "general")).strip()
        reason = str(intent_data.get("reason", "")).strip() or "Sin explicación adicional."

        return {"category": category, "reason": reason}   