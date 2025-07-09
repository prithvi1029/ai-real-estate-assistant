# FDS-to-TDS Generator Agentic System with Role-Based MAS, Feedback Loops, Guardrails, Retry, and RAG

import os
import re
import streamlit as st
from datetime import datetime
from PyPDF2 import PdfReader
from typing import TypedDict
from langchain_core.documents import Document
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch
from langchain.memory import ConversationBufferMemory
from pymongo import MongoClient

# --- MongoDB Logging ---
def get_collection(name):
    client = MongoClient("mongodb://localhost:27017")
    db = client["sap_fds_tds"]
    return db[name]

# --- API Keys ---
os.environ["TAVILY_API_KEY"] = "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"
os.environ["TOGETHER_API_KEY"] = "fa8e3ca9951bfafbd565a680b69625467935ee44b94e518e9eb6fbd706d0d937"

# --- LLM Agents Setup ---
llm_main = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                      openai_api_base="https://api.together.xyz/v1",
                      openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.5)

llm_score = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                       openai_api_base="https://api.together.xyz/v1",
                       openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.2)

llm_retry = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                       openai_api_base="https://api.together.xyz/v1",
                       openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.9)

memory = ConversationBufferMemory(return_messages=True)

# --- Define Agent State ---
class GraphState(TypedDict):
    question: str
    context: str
    web: str
    answer: str
    explanation: str
    score: str

# --- Vector Store Setup ---
def setup_vectordb_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    chunks = text.split("\n\n")
    docs = [Document(page_content=chunk.strip()) for chunk in chunks if chunk.strip()]
    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return DocArrayInMemorySearch.from_documents(docs, embed).as_retriever()

# --- LangGraph Star-MAS with Feedback, Retry, and Explanation ---
def build_graph(retriever):
    @RunnableLambda
    def retrieve_node(state: GraphState):
        docs = retriever.invoke(state["question"])
        return {"context": "\n".join([doc.page_content for doc in docs])}

    @RunnableLambda
    def web_search_node(state: GraphState):
        tavily = TavilySearch(api_key=os.environ["TAVILY_API_KEY"])
        return {"web": tavily.invoke(state["question"])}

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a professional SAP FDS-to-TDS Conversion Agent. Use safe, structured ABAP specs. Respond only in proper SAP TDS format. Use chain-of-thought reasoning. Add fallback sections if data is insufficient."),
        ("human", "FDS Description: {question}\n\nInternal Docs:\n{context}\n\nExternal Info:\n{web}\n\n→ TDS Output:")
    ])

    @RunnableLambda
    def answer_node(state: GraphState):
        memory.save_context({"input": state["question"]}, {"output": state.get("answer", "")})
        return {"answer": (answer_prompt | llm_main).invoke(state).content.strip()}

    explanation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're the Synthesizer Agent. Explain the logic behind the generated TDS, referencing both internal context and external sources."),
        ("human", "Generated TDS: {answer}\n\nDocs: {context}\n\nWeb Info: {web}")
    ])

    @RunnableLambda
    def explain_node(state: GraphState):
        return {"explanation": (explanation_prompt | llm_main).invoke(state).content.strip()}

    score_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're the QA Scoring Agent. Rate the TDS answer from 1-10. Justify if less than 7. If score <7, recommend retry."),
        ("human", "FDS: {question}\n\nTDS: {answer}\n\nExplanation: {explanation}")
    ])

    @RunnableLambda
    def score_node(state: GraphState):
        score_response = (score_prompt | llm_score).invoke(state).content
        match = re.search(r"\b(\d{1,2})\b", score_response)
        score = match.group(1) if match else "7"
        if int(score) < 7:
            retry_result = (answer_prompt | llm_retry).invoke(state).content.strip()
            return {"score": score, "answer": retry_result + "\n\n🔁 Retried for Quality <7"}
        return {"score": score}

    # Graph structure
    builder = StateGraph(GraphState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("web", web_search_node)
    builder.add_node("answer", answer_node)
    builder.add_node("explain", explain_node)
    builder.add_node("score", score_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "web")
    builder.add_edge("web", "answer")
    builder.add_edge("answer", "explain")
    builder.add_edge("explain", "score")
    builder.add_edge("score", END)

    return builder.compile()

# --- Streamlit UI ---
st.set_page_config("SAP FDS → TDS Generator")
st.title("🧩 SAP FDS to TDS Generator with Agentic AI")

log_collection = get_collection("conversion_logs")
file = st.file_uploader("📄 Upload FDS Document (PDF)", type="pdf")

if file:
    try:
        retriever = setup_vectordb_from_pdf(file)
        graph = build_graph(retriever)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        query = st.chat_input("💬 Ask to convert FDS to TDS...")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            with st.chat_message("assistant"):
                with st.spinner("🧠 Agent Generating TDS..."):
                    result = graph.invoke({"question": query})
                    log_data = {"timestamp": datetime.utcnow(), **result, "question": query}
                    log_collection.insert_one(log_data)

                    response = (
                        f"### ✅ Final TDS Output\n{result['answer']}\n\n"
                        f"---\n### 💡 Reasoning\n{result['explanation']}\n\n"
                        f"📊 **Confidence Score**: {result['score']}"
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.markdown(response)

    except Exception as e:
        st.error("❌ Failed to process the document.")
        st.exception(e)
