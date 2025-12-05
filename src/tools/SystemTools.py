from langchain_core.tools import tool
from src.functions.SystemFunctions import SystemFunctions
import json


def build_functions_tools(functions: SystemFunctions):
    """
    Construye una lista de tools de LangChain a partir de las operaciones
    principales del Systemfunctions. Estos tools serán usados por el agente
    orquestador.
    """

    @tool
    def index_documents() -> str:
        """
        Indexa todos los documentos TXT en el índice FAISS.
        Usa internamente el IndexerAgent del SystemFunctions.
        """
        functions.run_indexing()
        return "Indexación completada correctamente."


    @tool
    def classify_query(query: str) -> str:
        """
        Clasifica la intención de la consulta del usuario usando el ClassifierAgent.

        Devuelve un JSON con las claves:
        - "category": una de "busqueda", "resumen", "comparacion", "general" o "Error"
        - "reason": breve explicación de la decisión
        """
        intent = functions.run_classification(query)
        # Lo devolvemos como JSON para facilitar su uso
        return json.dumps(intent, ensure_ascii=False)

    @tool
    def retrieve_documents(query: str) -> str:
        """
        Recupera documentos relevantes para la consulta del usuario usando
        el RetrieverAgent.

        Devuelve un texto con los documentos encontrados, incluyendo:
        - índice [n]
        - ruta del archivo origen
        - un pequeño fragmento (snippet) del contenido.
        """
        docs = functions.run_retrieval(query, use_llm=False)

        if not docs:
            return "No se encontraron documentos relevantes."

        partes = []
        for i, d in enumerate(docs, start=1):
            source = d.metadata.get("source", "desconocido")
            snippet = d.page_content.replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            partes.append(f"[{i}] {source} -> {snippet}")

        return "\n".join(partes)

    @tool
    def answer_with_rag(query: str) -> str:
        """
        Genera una respuesta usando el agente RAG del SystemPipeline.

        Devuelve un JSON con:
        - "answer": respuesta generada en español (con citas [n])
        - "context": contexto numerado utilizado para generar la respuesta
        """
        result = functions.run_rag_respose(query, use_llm=False)
        # result es un dict: {"answer": ..., "context": ...}
        return json.dumps(result, ensure_ascii=False)

    @tool
    def evaluate_answer(query: str, rag_result_json: str) -> str:
        """
        Evalúa una respuesta generada por el RAG usando el EvaluatorAgent.

        Parámetros:
        - query: pregunta original del usuario.
        - rag_result_json: JSON devuelto por answer_with_rag, con:
            { "answer": "...", "context": "..." }

        Devuelve un JSON con:
        - "veredicto": "APROBAR" o "RECHAZAR"
        - "explicacion": breve explicación de la decisión.
        """

        try:
            rag_result = json.loads(rag_result_json)
        except json.JSONDecodeError:
            # Si viene algo raro, rechazamos por seguridad
            evaluation = {
                "veredicto": "RECHAZAR",
                "explicacion": "El JSON de la respuesta RAG es inválido o no se pudo parsear.",
            }
            return json.dumps(evaluation, ensure_ascii=False)
        
        answer = str(rag_result.get("answer", "")).strip()
        context = str(rag_result.get("context", "")).strip()

        evaluation = functions.run_evaluation(query, context, answer)
        return json.dumps(evaluation, ensure_ascii=False)

    # Lista de AL MENOS 5 tools
    return [
        index_documents,
        classify_query,
        retrieve_documents,
        answer_with_rag,
        evaluate_answer,
    ]
