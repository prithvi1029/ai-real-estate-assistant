# 🧪 SAP Test Case Generator AI Agent

## Overview

This project implements an AI-powered agent that generates edge-to-edge test cases from uploaded SAP FDS (Functional Design Specifications) or TDS (Technical Design Specifications) documents. It leverages Retrieval-Augmented Generation (RAG), web search, conversational memory, feedback loops, MongoDB logging, and a LangGraph multi-agent pipeline to ensure robust, accurate, and explainable outputs.

---

## 🔧 Features

* 🔍 **Document-based Retrieval**: Vector store (HuggingFace + DocArrayInMemorySearch) based PDF chunk retrieval.
* 🌐 **Web Search Integration**: Using Tavily API for external signal augmentation.
* 💬 **Conversational Memory**: Retains interaction history with `ConversationBufferMemory`.
* 🔁 **Feedback Loop**: Self-scoring + answer retry if score < 7.
* 🧠 **Multi-Agent Workflow**: LangGraph pipeline with nodes for retrieval, answer generation, explanation, scoring.
* 📊 **Logging**: MongoDB session logging for traceability.
* 🎛️ **Streamlit UI**: Chat interface for PDF upload and prompt-based interaction.

---

## 🛠️ Setup Instructions

### 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Required API Keys (Set as Environment Variables)

* `TOGETHER_API_KEY` – for Mixtral model via Together
* `TAVILY_API_KEY` – for web search

You can export them in your shell or `.env`:

```bash
export TOGETHER_API_KEY="<your-key>"
export TAVILY_API_KEY="<your-key>"
```

### 3. MongoDB

* Start local instance at `mongodb://localhost:27017`
* Create database `sap_testcase_gen` and collection `session_logs`

### 4. Run the App

```bash
streamlit run test_case_generating_agent.py
```

---

## 📄 Sample Prompts

* "Generate test cases for the Purchase Order creation flow."
* "Create edge scenarios for invoice verification failure."
* "What are the test cases for incorrect tax logic in FDS?"

---

## 📚 Techniques Implemented

### ✅ From Research:

* **Corrective RAG**: Retry answer if feedback score < 7
* **Self RAG**: Explains reasoning of output
* **Adaptive RAG**: Temperature tuning based on score feedback
* **LangGraph Orchestration**: StateGraph pipeline with edge routing
* **Web-Augmented RAG**: Uses Tavily for external augmentation
* **Memory-Augmented Chat**: Maintains dialogue context

---

## 🧩 Agent Pipeline Architecture

```
Upload PDF
   |
[Retriever] <-----
   |             |
[Web Search]    [Memory Save]
   |             |
[LLM Answer Generation]
   | --> [LLM Explanation Node]
            |
     [Scoring Agent (LLM)]
        |
 [Retry if score < 7]
        |
       END
```

---

## 📁 File Structure

```
├── test_case_generating_agent.py  # Streamlit app & LangGraph agent
├── requirements.txt               # Python dependencies
├── README.md                      # This file
```

---

## 📌 Future Enhancements

* ✅ Agent scoring visualization in UI
* ✅ Role-based prompt routing
* ✅ Tabular document RAG with layout-aware retrieval
* ✅ Integration with other SAP agents (FDS ↔ TDS ↔ Test Case ↔ ABAP)
* ✅ Prompt suggestions as UI buttons

---

## 👨‍💻 Author

Developed by \[Your Name / Team] as part of an intelligent SAP automation suite using LangGraph and Agentic AI principles.

---

## 📜 License

MIT License
