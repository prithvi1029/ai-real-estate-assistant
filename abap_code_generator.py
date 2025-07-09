# Final ABAP Code Assistant System with Star MAS + Specialized Agents + Scoring + Feedback Loops + Prompt Personalization + Guardrails + Conversational Memory

import os
import re
import traceback
import streamlit as st
from PyPDF2 import PdfReader
from datetime import datetime
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

# Mongo helper setup
def get_collection(collection_name):
    client = MongoClient("mongodb://localhost:27017")
    db = client["abap_assistant"]
    return db[collection_name]

# API Keys
os.environ["TAVILY_API_KEY"] = "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"
os.environ["TOGETHER_API_KEY"] = "fa8e3ca9951bfafbd565a680b69625467935ee44b94e518e9eb6fbd706d0d937"

# LLMs
llm_main = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.5)
llm_score = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.2)
llm_retry = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.9)

# Conversational memory setup
memory = ConversationBufferMemory(return_messages=True)

class GraphState(TypedDict):
    question: str
    context: str
    web: str
    answer: str
    explanation: str
    score: str

# Vector DB setup from PDF
def setup_vectordb_from_pdf(file):
    reader = PdfReader(file)
    text = "\n".join([p.extract_text() or "" for p in reader.pages])
    chunks = text.split("\n\n")
    docs = [Document(page_content=chunk.strip()) for chunk in chunks if chunk.strip()]
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return DocArrayInMemorySearch.from_documents(docs, embed_model).as_retriever()

# LangGraph Build (Star Architecture with Specialized Roles)
def build_graph(retriever):
    @RunnableLambda
    def retrieve_node(state: GraphState):
        docs = retriever.invoke(state["question"])
        return {"context": "\n".join([d.page_content for d in docs])}

    @RunnableLambda
    def web_search_node(state: GraphState):
        search_tool = TavilySearch(api_key=os.environ["TAVILY_API_KEY"], max_results=3)
        result = search_tool.invoke(state["question"])
        return {"web": result}

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a role-specialized ABAP RICEF generator. Keep it simple and safe. Use logic, professional tone, and chain-of-thought reasoning. Adapt to user persona. Avoid hallucination. Only respond to relevant ABAP topics."),
        ("human", "Sender: User\nMessage: {question}\nReceiver: ABAP Assistant\nStyle: professional\nReasoning: explain every step logically.\nContext: {context}\nWeb: {web}")
    ])

    @RunnableLambda
    def answer_node(state: GraphState):
        memory.save_context({"input": state["question"]}, {"output": state.get("answer", "")})
        return {"answer": (answer_prompt | llm_main).invoke(state).content.strip()}

    explanation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're the Synthesizer Agent. Provide reasoning for the ABAP output. Keep user in the loop and highlight safety of each suggestion."),
        ("human", "Answer: {answer}\n\nContext: {context}\n\nWeb: {web}")
    ])

    @RunnableLambda
    def explain_node(state: GraphState):
        return {"explanation": (explanation_prompt | llm_main).invoke(state).content.strip()}

    score_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're the Scoring Agent. Rate from 1–10. If <7, suggest improvements. Apply guardrails for impersonation or invalid logic."),
        ("human", "Question: {question}\nAnswer: {answer}\nExplanation: {explanation}")
    ])

    @RunnableLambda
    def score_node(state: GraphState):
        result = (score_prompt | llm_score).invoke(state).content
        match = re.search(r"\b(\d{1,2})\b", result)
        score = match.group(1) if match else "7"
        if int(score) < 7:
            improved = (answer_prompt | llm_retry).invoke(state).content.strip()
            return {"score": score, "answer": improved + f"\n\n[Refined by feedback loop]"}
        return {"score": score}

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

# UI
st.set_page_config("ABAP Code Assistant")
st.title("🤖 ABAP Code Assistant (MAS, Feedback, Tabular, Web, Guardrails, Memory)")

log_collection = get_collection("session_logs")
file = st.file_uploader("Upload FDS or TDS PDF", type="pdf")

if file:
    try:
        retriever = setup_vectordb_from_pdf(file)
        graph = build_graph(retriever)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        query = st.chat_input("Ask about ABAP code, pseudocode, logic...")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            with st.chat_message("assistant"):
                with st.spinner("Working with MAS agents..."):
                    result = graph.invoke({"question": query})

                    log_data = {"timestamp": datetime.utcnow(), **result, "question": query}
                    log_collection.insert_one(log_data)

                    output = f"### ✅ Answer\n{result['answer']}\n\n---\n### 🧠 Explanation\n{result['explanation']}\n\n🎯 Score: {result['score']}"
                    st.session_state.chat_history.append({"role": "assistant", "content": output})
                    st.markdown(output)

    except Exception as e:
        st.error("Upload failed.")
        st.exception(e)
