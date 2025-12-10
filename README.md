# Sports-Multiagent-App
> **La Enciclopedia Deportiva Inteligente impulsada por Agentes Autónomos.**

[![LangChain](https://img.shields.io/badge/AI-LangChain_1.0-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.langchain.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-orange?style=for-the-badge)]()
[![Knowledge Base](https://img.shields.io/badge/Knowledge-100%2B_Artículos_Deportivos-success?style=for-the-badge)]()
[![Status](https://img.shields.io/badge/Status-Game_Ready-ff0000?style=for-the-badge)]()

---

## ¿De qué va este proyecto?

Esta aplicación implementa un **sistema multi-agente** que actúa como una *enciclopedia deportiva inteligente*.

- Usa **LangChain 1.x** y **Gemini (Google Generative AI)**.
- Indexa documentos de deportes (fútbol, béisbol, ajedrez, etc.) en un índice vectorial **FAISS**.
- Expone un **bot de Telegram** que:
  - Recibe preguntas en lenguaje natural.
  - Recupera los fragmentos más relevantes.
  - Genera respuestas con **RAG** (Retrieval-Augmented Generation) con citas.
  - Permite **subir nuevos .txt o .pdf** para ampliar la base de conocimiento en caliente.

Bot de Telegram: **[@AgenticIALangchain_bot](https://t.me/AgenticIALangchain_bot)**  
*(primero debes ejecutar el proyecto en tu máquina)*

---

## Arquitectura de Agentes

El sistema está organizado en varios agentes especializados:

### 1. `IndexerAgent`
- Consume documentos `.txt` desde `data/raw_docs`.
- Los divide en *chunks* con `RecursiveCharacterTextSplitter`.
- Genera embeddings (Gemini) y construye/actualiza el índice **FAISS**.
- Soporta:
  - Creación inicial del índice.
  - Actualización incremental cuando se añade un nuevo documento.

### 2. `ClassifierAgent`
- Clasifica la **intención** de la consulta del usuario usando el LLM.
- Categorías de ejemplo: `busqueda`, `resumen`, `comparacion`, `general`.
- Devuelve un JSON con:
  - `category`
  - `reason` (explicación breve)

### 3. `RetrieverAgent`
- Carga el índice FAISS desde disco en cada consulta.
- Ejecuta **búsqueda de similitud semántica**.
- Puede opcionalmente reescribir la query con el LLM (`rewrite_query`) para optimizar la recuperación.

### 4. `RagAgent`
- Combina:
  - la **pregunta del usuario**, y
  - los **fragmentos recuperados** por el `RetrieverAgent`.
- Construye un contexto numerado `[1], [2], ...` con el origen de cada fragmento.
- Usa un `ChatPromptTemplate` para forzar:
  - respuesta en español,
  - uso exclusivo del contexto,
  - citas tipo `[n]` en la respuesta.

### 5. `EvaluatorAgent`
- Evalúa la respuesta generada por el RAG:
  - ¿Está respaldada por el contexto?
  - ¿Es coherente y clara?
  - ¿Responde a la pregunta?
- Devuelve un JSON con:
  - `veredicto`: `"APROBAR"` o `"RECHAZAR"`,
  - `explicacion`: breve justificación.

### 6. `OrchestratorAgent`
- Agente principal que **orquesta** el flujo usando *tools* de LangChain:
  - `index_documents`
  - `classify_query`
  - `retrieve_documents`
  - `answer_with_rag`
  - `evaluate_answer`
- Decide cuándo llamar a cada tool y genera la **respuesta final** al usuario.

### 7. `SystemFunctions` + `SystemTools`
- `SystemFunctions`: clase que centraliza las operaciones de alto nivel (indexar, clasificar, recuperar, responder, evaluar) usando los agentes internos.
- `SystemTools`: convierte esas funciones en **tools LangChain** para que el `OrchestratorAgent` pueda invocarlas de forma autónoma.

---

## Estructura de Carpetas

```bash
.
├─ data/
│  ├─ raw_docs/        # Documentos base en TXT (y nuevos TXT tras conversión)
│  ├─ new_docs/        # Archivos recién subidos por Telegram (PDF/TXT temporales)
│  └─ faiss_index/     # Índice FAISS (index.faiss + index.pkl)
│
├─ src/
│  ├─ agents/
│  │  ├─ indexer_agent.py
│  │  ├─ classifier_agent.py
│  │  ├─ retriever_agent.py
│  │  ├─ rag_agent.py
│  │  ├─ evaluator_agent.py
│  │  └─ orchestrator_agent.py
│  │
│  ├─ bot/
│  │  └─ telegram_bot.py        # Lógica del bot de Telegram
│  │
│  ├─ functions/
│  │  └─ SystemFunctions.py     # Puente entre agentes y tools
│  │
│  ├─ tools/
│  │  └─ SystemTools.py         # Definición de tools para LangChain
│  │
│  └─ utils/
│     ├─ data_extraction.py     # Descarga/creación de documentos base
│     └─ pdf_to_text.py         # Conversión PDF → TXT y movimiento de archivos
│
├─ main.py                      # Punto de entrada del proyecto
├─ requirements.txt
└─ README.md
```
---
## Primer uso -  Desde la raiz del repositorio
  1. Crear ambiente para instalar dependencias

  ```bash
  python -m venv .venv (python puede varias a python3 o similares si se usa linux o mac)
  .venv\Scripts\activate
  ```

  2. Instalar dependencias
   ```bash
  pip install -r requirements.txt
  ```

  3. Prender el bot
  ```bash
  python main.py
  ```
  4. Iniciar el bot
  Desde telegram, escribir /start a @AgenticIALangchain_bot

  5. Preguntarle al bot 
  
