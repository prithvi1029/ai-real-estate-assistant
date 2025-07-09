# 🏡 Real Estate AI Assistant

An AI-powered real estate assistant built with LangChain, LangGraph, and Streamlit. This application allows users to upload apartment listing CSVs and ask natural language questions about listings. The assistant responds using a multi-agent system that combines structured CSV data, semantic search (RAG), web search, explanation, scoring, and retry logic.

---

## 📦 Features

### ✅ CSV Upload and Vectorization
- Upload your apartment listings in CSV format.
- CSV rows are enriched and embedded using HuggingFace `MiniLM` embeddings.
- Stored in memory using `DocArrayInMemorySearch` for fast retrieval.

### 🧠 Multi-Agent Workflow (LangGraph)
- **Query Transformation**: Reformulates user query into a structured search-friendly prompt.
- **CSV Matching Agent**: Retrieves top matching apartments from vectorstore.
- **RAG Agent**: Performs semantic answer generation from listings using LLM.
- **Web Search Agent**: Fetches external context using Tavily API.
- **Fusion Synthesizer**: Combines outputs of all sources to provide a clear answer.
- **Explanation Agent**: Generates a reasoning-based explanation of the final answer.
- **Scoring Agent**: Evaluates final answer using a scoring LLM and provides a 1-10 rating.

### 🔁 Feedback & Retry Loops
- If score < 7, the system performs an **adaptive retry** to regenerate better answers.
- Implements a **voting mechanism** between original and improved answers to ensure the best response.

### 📝 Logging
- Stores full session logs to MongoDB using `mongo_helper.get_collection()`.

---



### 📝 Graph structure:

           +-----------+             
           | __start__ |             
           +-----------+             
                  *                  
                  *                  
                  *                  
           +-----------+             
           | transform |             
           +-----------+             
          **      *     **           
        **        *       **         
      **          *         **       
+-----+       +-----+       +-----+  
| csv |*      | rag |       | web |  
+-----+ **    +-----+     **+-----+  
          **      *     **           
            **    *   **             
              **  * **               
          +-------------+            
          | synthesizer |            
          +-------------+            
                  *                  
                  *                  
                  *                  
            +---------+              
            | explain |              
            +---------+              
                  *                  
                  *                  
                  *                  
             +-------+               
             | score |               
             +-------+               
                  *                  
                  *                  
                  *                  
            +---------+              
            | __end__ |              
            +---------+              
## 🚀 How to Run

1. Clone this repository or save the script.
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Launch the Streamlit app:
```bash
streamlit run final_streamlit_real_estate_assistant.py
```

---

## 🧪 Example Query
> “Show me 3-bedroom apartments in Warsaw under 5000 PLN with parking.”

### Output Includes:
- 🏠 Final Answer with Green/Amber/Red rating.
- 🧠 Reasoning behind the ranking.
- 🎯 Score out of 10.
- 🔁 Refined output if original score was low.

---

## 📚 Techniques Used

- **LangGraph**: State machine for multi-agent task routing.
- **Corrective RAG**: Retry logic triggered based on explanation + scoring.
- **Self-RAG**: Explanation & answer improvement driven by self-evaluation.
- **Adaptive RAG**: Dynamic routing of retry and answer fusion.
- **Voting Agents**: Choose best answer using an evaluator LLM.
- **Tabular RAG**: Embedding-based matching on tabular data (CSV).
- **MongoDB Logging**: For analysis and dashboarding.

---

## 📁 Folder Structure

```
.
├── final_streamlit_real_estate_assistant.py
├── utils/
│   └── mongo_helper.py
├── data/
│   └── example_listings.csv
├── requirements.txt
└── README.md
```

---

## 🔐 API Keys

- `TOGETHER_API_KEY` for LLM (e.g., Mixtral-8x7B via Together.ai)
- `TAVILY_API_KEY` for Web Search (Tavily)

Add them to environment variables or `.env` file.

---

## 🙌 Acknowledgements

- LangChain + LangGraph for orchestration
- Tavily Search API for web augmentation
- HuggingFace Embeddings for semantic vector search
- Mixtral-8x7B for all LLM tasks

---

## 📬 Contact

Feel free to reach out for any support or collaboration!