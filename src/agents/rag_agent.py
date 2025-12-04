from langchain_core.language_models.chat_models import BaseChatModel
from src.agents.retriever_agent import RetrieverAgent
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

class RagAgent:
    """
    Agente RAG (Retrieval-Augmented Generation).
    Se encarga de:
        - Recibir una consulta del usuario
        - Recuperar documentos relevantes usando el RetrieverAgent
        - Generar una respuesta basada en los documentos recuperados
    """

    # Constructor
    def __init__(self, retriever_agent: RetrieverAgent, llm: BaseChatModel):
        """
        Parámetros:
        - retriever_agent: Instancia del agente de recuperación
        - llm: Modelo de lenguaje para generación de respuestas
        """
        self.retriever_agent = retriever_agent
        self.llm = llm


    def get_context(self, query: str, use_llm: bool = True) -> str:
        # Recuperar documentos relevantes
        documents = self.retriever_agent.retrieve_documents(query, use_llm=use_llm)

        partes = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "desconocido")
            filename = Path(source).name

            texto = doc.page_content.replace("\n", " ")
            # Si quieres, recortar un poco:
            if len(texto) > 800:
                texto = texto[:800] + "..."

            partes.append(f"[{i}] ({filename}): {texto}")

        context = "\n\n".join(partes)
        return context
    

    prompt_template = ChatPromptTemplate.from_template("""
    Eres un asistente experto en comprensión de documentos.

    Dispones de varios fragmentos numerados de contexto sobre un tema.
    Cada fragmento está etiquetado entre corchetes, por ejemplo [1], [2], etc.
    Tu tarea es responder SIEMPRE en español, usando EXCLUSIVAMENTE
    la información que aparece en esos fragmentos.

    Instrucciones para las citas:
    - Cuando uses información de un fragmento, añade al final de la frase
    la cita correspondiente, por ejemplo [1] o [2].
    - Si una idea se basa en varios fragmentos, puedes usar varias citas, por ejemplo [1][3].
    - No inventes citas ni números que no aparezcan en el contexto.
    - Si el contexto no contiene suficiente información para responder algo,
    dilo explícitamente (por ejemplo: "En los fragmentos proporcionados no se menciona X").

    Contexto:
    {context}

    Pregunta del usuario:
    \"\"\"{question}\"\"\"

    Responde de forma clara, estructurada y en español,
    incluyendo las citas [n] apropiadas.
    """)

    # Generar respuesta basada en documentos recuperados
    def generate_response(self, query: str) -> str:
        context = self.get_context(query, use_llm=False)
        messages = self.prompt_template.invoke({
            "context": context,
            "question": query,
        })
        response = self.llm.invoke(messages)
        return response.content.strip()