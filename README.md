# 🏟️ Sports-Multiagent-App
> **La Enciclopedia Deportiva Inteligente impulsada por Agentes Autónomos.**

[![LangChain](https://img.shields.io/badge/AI-LangChain_1.0-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.langchain.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-orange?style=for-the-badge)]()
[![Knowledge Base](https://img.shields.io/badge/Knowledge-100_Sports-success?style=for-the-badge&logo=googledocs&logoColor=white)]()
[![Status](https://img.shields.io/badge/Status-Game_Ready-ff0000?style=for-the-badge)]()

---

## ⚡ ¿De qué trata?

Imagina tener a un experto olímpico en tu bolsillo que conoce las reglas, la historia y los detalles técnicos de **100 deportes diferentes**.

**Sports-Multiagent-app** no es un simple chatbot. Es un sistema de **Agentic AI** (Inteligencia Artificial Agéntica) que utiliza una arquitectura RAG (Retrieval-Augmented Generation) para consultar una base de datos curada de 100 documentos especializados.

Desde lo más popular como el **Fútbol** ⚽ y el **Baloncesto** 🏀, hasta disciplinas específicas como el **Kitesurf** 🪁, la **Espeleología deportiva** 🧗 o el **Ultimate Frisbee** 🥏.

---

## 🧠 La Arquitectura: "El Equipo"

El sistema funciona como un cuerpo técnico deportivo, donde diferentes agentes colaboran para dar la respuesta perfecta.

| Agente (Rol) | Misión | Tecnología |
| :--- | :--- | :--- |
| **👮 El Árbitro (Router)** | Analiza la pregunta del usuario e identifica de qué deporte se está hablando. Redirige el flujo al documento correcto. | `LangChain RouterChain` |
| **🕵️ El Scout (Retriever)** | Busca en la base de datos vectorial (VectorStore) la información exacta dentro del documento específico (ej: `Sumo.txt`). | `ChromaDB` / `FAISS` |
| **🎙️ El Comentarista (Answerer)** | Toma la información cruda y genera una respuesta natural, educativa y precisa para el usuario. | `OpenAI GPT-4` / `Llama 3` |

---

## 📚 El Dataset: "The 100 Challenge"

El corazón de este proyecto es su base de conocimiento. Hemos recopilado y procesado **100 documentos de texto plano**, cada uno dedicado exclusivamente a un deporte.

> **¿Por qué 100 documentos separados?**
> Para garantizar la **precisión**. Al aislar el contexto de cada deporte, evitamos que el agente confunda las reglas del *Rugby* con las del *Fútbol Americano*.

### 📂 Estructura del Conocimiento (`/data`)
```text
data/
 ├── 🏹 Archery.txt
 ├── 🏸 Badminton.txt
 ├── 🏏 Cricket.txt
 ├── ...
 ├── 🥋 Judo.txt
 ├── 🏄 Surfing.txt
 └── 🧘 Yoga.txt
 (Total: 100 archivos)
