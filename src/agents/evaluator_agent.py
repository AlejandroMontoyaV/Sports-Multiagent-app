from langchain_core.language_models.chat_models import BaseChatModel
import re
import json
from typing import Dict
class EvaluatorAgent:
    """
    Agente Verificador / Crítico (Evaluador).

    Se encarga de:
      - Evaluar si la respuesta generada por el agente RAG:
          * Está respaldada por el contexto recuperado.
          * Es coherente y clara.
          * Responde a la pregunta del usuario.
      - Producir un veredicto (APROBAR / RECHAZAR) junto con una explicación.

    Este agente utiliza un LLM para realizar el razonamiento y la validación.
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def evaluate_response(self, question: str, context: str, answer: str) -> Dict[str, str]:
        prompt = f"""
        Eres un evaluador experto en respuestas generadas por modelos de lenguaje.

        Tienes la siguiente pregunta del usuario:
        \"\"\"{question}\"\"\"

        Y la siguiente respuesta generada:
        \"\"\"{answer}\"\"\"

        Además, tienes este contexto recuperado:
        \"\"\"{context}\"\"\"

        Tu tarea es evaluar si la respuesta está respaldada por el contexto,
        es coherente y clara, y responde adecuadamente a la pregunta del usuario.

        Devuelve un veredicto en formato JSON con las siguientes claves:
        - "veredicto": "APROBAR" o "RECHAZAR"
        - "explicacion": breve explicación (1-3 frases) de tu decisión

        Recuerda:
        - Si la respuesta no está respaldada por el contexto, RECHAZAR.
        - Si la respuesta es confusa o no responde a la pregunta, RECHAZAR.
        - Si todo está bien, APROBAR.
        """.strip()

        response = self.llm.invoke(prompt)
        # Función para extraer JSON del texto(el llm lo daba rarito)
        def extract_json(text):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return match.group(0) if match else None
        json_str = extract_json(response.content)

        # Se devuelve el resultado como diccionario
        if json_str:
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                pass
        
        return {
            "veredicto": "RECHAZAR",
            "explicacion": "No se pudo extraer un veredicto válido del LLM."
        }