# ABAP Code Assistant (Multi-Agent AI System)

This project is an advanced, AI-powered ABAP Code Assistant for generating and explaining ABAP code, pseudocode, and logic using Functional Design Specifications (FDS) or Technical Design Specifications (TDS) in PDF format.

---

## 🔧 Features

- 🧠 Multi-Agent System (MAS) with Star Architecture
- 📚 Document-based Retrieval using Vector DB
- 🌐 Web Search using Tavily
- 🔍 Role-based Prompting and System Message Personalization
- 🧪 Feedback Loops with Score-based Refinement
- 🎯 Self-Scoring with Guardrails
- 🗂️ Tabular Document Support
- 🕹️ Real-Time Streamlit UI
- 💬 MongoDB Logging for Conversations

---

## 🧱 Architecture Overview

```
             +--------------------+
             |  User Query Input  |
             +--------------------+
                       |
                       v
            +---------------------+
            |    Retriever Node   | ←-- Embedding-based PDF Retrieval
            +---------------------+
                       |
                       v
            +---------------------+
            |   Web Search Node   | ←-- Tavily Web API
            +---------------------+
                       |
                       v
            +---------------------+
            |    Answer Agent     | ←-- Role-Specialized ABAP Generator
            +---------------------+
                       |
                       v
            +---------------------+
            |  Explanation Agent  | ←-- Synthesizer with CoT Reasoning
            +---------------------+
                       |
                       v
            +---------------------+
            |     Score Agent     | ←-- Self-Scoring + Feedback Loop
            +---------------------+
                       |
                       v
            +---------------------+
            |   Output Display    |
            +---------------------+
```

---

## 📁 File Structure

- `code_assist.py`: Main Streamlit app
- `utils/`: Helpers (MongoDB, RAG, embeddings)
- `requirements.txt`: Dependencies
- `README.md`: Project guide

---

## ▶️ How to Run

```bash
# 1. Create venv
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit app
streamlit run code_assist.py
```

---

## 🔑 Environment Variables

Ensure the following keys are set:

```bash
export TOGETHER_API_KEY=your_together_api_key
export TAVILY_API_KEY=your_tavily_api_key
```

---

## 🧠 Sample Questions

- Generate ABAP code for a sales order report
- Explain pseudocode from uploaded FDS
- Create a loop for table updates with validations
- Search web for best ABAP practices in module pool programming

---

## 📌 Research Techniques Used

- **Adaptive RAG**: Adjusts agent pathways dynamically
- **Corrective RAG**: Retries if output score is low
- **Self-RAG**: Explanation and scoring agents ensure transparency
- **Feedback Loops**: Trigger re-generation if < 7
- **Prompt Personalization**: Role/Receiver-based conditioning
- **Star Topology MAS**: Central orchestrator with specialized agents
- **Guardrails**: Score agent checks for hallucinations or impersonation
- **Path Dependence**: Maintains logical progression of response refinement

---

## 📸 Architecture Diagram

![Architecture Diagram](architecture_diagram.png)

---

## 🧾 License

MIT License. Feel free to modify and extend.

---

## 🤝 Contributors

Built by Abhishek Prithvi Teja | Powered by LangGraph + Mixtral + LangChain + HuggingFace
---

## 🧭 Architecture Diagram

Below is the high-level architecture of the ABAP Code Assistant:

```
                   +--------------------+
                   |   User Interface   |
                   |  (Streamlit Chat)  |
                   +---------+----------+
                             |
                             v
                +------------+------------+
                |  LangGraph MAS Pipeline |
                | (Star Topology Workflow)|
                +------------+------------+
                             |
        +---------+----------+----------+----------+
        |         |                     |          |
        v         v                     v          v
   Retriever   Web Search        Answer Generator    (PDF Vector) (Tavily API)      (Role-Prompted LLM)                                                      Feedback Loop
                                                     /
        +---------+----------+----------+----------+
                             |
                             v
                      Explanation Agent
                             |
                             v
                       Scoring Agent
                       (Guardrails)
                             |
                             v
                        MongoDB Logging

```

> 🧩 Each agent (Retriever, Web Search, Answer, Explanation, Scorer) is role-specialized and works in parallel for efficiency and feedback-based optimization.
