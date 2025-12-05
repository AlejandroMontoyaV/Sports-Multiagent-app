from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from typing import List


# Aqui todos los demás agentes se van a considerar como tools
class OrchestratorAgent:
    """
    Agente orquestador del sistema.

    - Recibe la pregunta del usuario.
    - Decide qué tools llamar (clasificador, RAG, evaluador, etc.).
    - Usa las respuestas de los tools para construir una respuesta final en español.
    """

    def __init__(self, llm: BaseChatModel, tools: List[BaseTool]):
        """
        Parámetros
        ----------
        llm : BaseChatModel
            Modelo de lenguaje que actuará como agente orquestador.
        tools : List[BaseTool]
            Lista de tools de LangChain disponibles (index_documents, classify_query, etc.).
        """
        self.llm = llm
        self.tools = tools

        # Prompt del agente orquestador
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
                """
            Eres el AGENTE ORQUESTADOR de un sistema de pregunta-respuesta sobre DEPORTES.

            Dispones de varias herramientas (tools) que puedes invocar cuando lo necesites.
            Tu objetivo es ayudar al usuario respondiendo SIEMPRE en español de forma clara
            y correcta, utilizando las herramientas de manera razonada.

            Herramientas principales (resumen):
            - index_documents(): indexa los documentos en el índice FAISS (úsala al inicio o si el índice no está actualizado). Esta solo se ejecuta si se añaden nueos documentos
            - classify_query(query): clasifica la intención de la consulta del usuario. Las respuestas van a ser "busqueda", "resumen", "comparación" ó "general"
            - retrieve_documents(query): obtiene fragmentos relevantes desde los documentos.
            - answer_with_rag(query): genera una respuesta basada en los documentos recuperados (RAG). Devuelve un JSON con "answer" y "context".
            - evaluate_answer(query, rag_result_json): evalúa la respuesta del RAG usando el contexto y devuelve un JSON con "veredicto" y "explicacion".

            Flujo recomendado:
            1. Para una nueva consulta del usuario:
            - Usa primero classify_query(query) para entender el tipo de pregunta.
            2. Si la consulta requiere información basada en documentos("busqueda", "resumen", "comparación"):
            - Llama a answer_with_rag(query) para obtener una respuesta propuesta.
            - A continuación, llama a evaluate_answer(query, rag_result_json) con el JSON que devolvió answer_with_rag.
            - Si el veredicto es "RECHAZAR", puedes intentar UNA segunda vez con answer_with_rag
                y volver a evaluar. Si aún así sale "RECHAZAR", explícale al usuario que no puedes
                dar una respuesta fiable basada en los documentos.
            3. Si la respuesta no requiere información basada en documentos ("general"), responde con lo que te de el LLM
            4. Solo cuando estés satisfecho con la información de las herramientas,
            responde al usuario de forma directa y clara.

            Muy importante:
            - NO inventes herramientas nuevas: usa solo las que están disponibles.
            - NO hagas suposiciones fuertes sin apoyo de los tools.
            - La respuesta final al usuario debe ser un texto en español, sin JSON.
            - Muestra las ideas de manera ordenada y, si es útil, menciona de forma natural
            que tu respuesta está basada en los documentos del sistema.
                            """.strip()
            ),
            # Mensaje del usuario
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        # Creamos el agente de tool-calling y su ejecutor
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=False, return_intermediate_steps=True)

    def run(self, query: str) -> str:
        """
        Ejecuta el agente orquestador con la consulta del usuario.
        """
        result = self.executor.invoke({"input": query})
        pasos = result.get("intermediate_steps", [])

        print("\n🔍 Pasos intermedios:")
        for paso in pasos:
            tool_call, tool_output = paso
            print(f"- Tool: {tool_call.tool}, args: {tool_call.tool_input}")
            print(f"  Output: {tool_output}\n")
        
        return result.get("output", "")