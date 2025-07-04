# streamlit_app_upload.py

import os
import traceback
import streamlit as st
import pandas as pd
from typing import TypedDict
from langchain_core.documents import Document
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch

# 🔐 API Keys
os.environ["TAVILY_API_KEY"] = "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"
os.environ["TOGETHER_API_KEY"] = "fa8e3ca9951bfafbd565a680b69625467935ee44b94e518e9eb6fbd706d0d937"

# 🧠 LLMs
llm_transform = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.7)
llm_main = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.5)
llm_retry = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.9)
llm_score = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.3)

# ✅ State
class GraphState(TypedDict):
    question: str
    transformed_question: str
    rows: list
    rag: str
    web: str
    answer: str
    explanation: str
    score: str

def enrich_row(row):
    return f"{row['rooms']} room apartment in {row['city']} priced at {row['price']} PLN. Area: {row['squareMeters']} sqm. Parking: {row['hasParkingSpace']}. Balcony: {row['hasBalcony']}. Elevator: {row['hasElevator']}. Condition: {row['condition']}."

def setup_vectordb_from_df(df):
    df = df.dropna()
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = [Document(page_content=enrich_row(row)) for _, row in df.iterrows()]
    vstore = DocArrayInMemorySearch.from_documents(docs, embed_model)
    return vstore.as_retriever(search_type="mmr")

def build_graph(retriever):
    query_transform_prompt = ChatPromptTemplate.from_messages([
        ("system", "Reformulate user queries to match real estate listing style."),
        ("human", "Original Query: {question}")
    ])

    @RunnableLambda
    def query_transform_node(state: GraphState):
        result = (query_transform_prompt | llm_transform).invoke({"question": state["question"]})
        return {"transformed_question": result.content.strip()}

    @RunnableLambda
    def csv_row_retrieve_node(state: GraphState):
        rows = retriever.invoke(state["transformed_question"])
        return {"rows": [r.page_content for r in rows]}

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user based on retrieved apartment data."),
        ("human", "{context}\nQuestion: {question}")
    ])
    @RunnableLambda
    def semantic_rag_node(state: GraphState):
        docs = retriever.invoke(state["transformed_question"])
        rag_input = {
            "context": "\n".join([doc.page_content for doc in docs]),
            "question": state["transformed_question"]
        }
        result = (rag_prompt | llm_main).invoke(rag_input)
        return {"rag": result.content.strip()}

    search_tool = TavilySearch(api_key=os.environ["TAVILY_API_KEY"], max_results=3)
    @RunnableLambda
    def web_search_node(state: GraphState):
        content = search_tool.invoke(state["transformed_question"] + " in Poland")
        return {"web": content}

    fusion_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a helpful real estate assistant using CSV and RAG. Prioritize listings."),
        ("human", """You're an AI real estate assistant. Use CSV rows as primary source.
Rank apartments as Green (best match), Amber (partial match), or Red (weak match) based on:
- Budget fit
- Room count
- Parking availability
- City match

Use RAG and Web Search as supplemental info.
Question: {question}
CSV Rows:\n{rows}
RAG:\n{rag}
Web:\n{web}
Provide a ranked, clear answer:""")
    ])
    @RunnableLambda
    def summarize_fusion_node(state: GraphState):
        fusion_input = {
            "question": state["question"],
            "rows": "\n".join(state.get("rows", [])[:5]),
            "rag": state["rag"],
            "web": state["web"]
        }
        result = (fusion_prompt | llm_main).invoke(fusion_input)
        return {"answer": result.content.strip()}

    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a reasoning analyst. Explain how the answer was derived."),
        ("human", "RAG: {rag}\n\nRows: {rows}\n\nWeb: {web}")
    ])
    @RunnableLambda
    def explain_node(state: GraphState):
        result = llm_main.invoke(explain_prompt.format(
            rag=state["rag"],
            rows="\n".join(state["rows"]),
            web=state["web"]
        ))
        return {"explanation": result.content.strip()}

    score_prompt = ChatPromptTemplate.from_messages([
        ("system", "Score from 1-10 how well the answer fits the user's query and cite a reason."),
        ("human", "Query: {question}\n\nAnswer: {answer}\n\nExplanation: {explanation}\n\nScore and reason:")
    ])
    @RunnableLambda
    def score_node(state: GraphState):
        res = llm_score.invoke(score_prompt.format(
            question=state["question"],
            answer=state["answer"],
            explanation=state["explanation"]
        )).content.strip()
        score, reason = res.split("\n", 1) if "\n" in res else (res, "No reason provided.")
        try:
            if int(score) < 7:
                retry_input = {
                    "question": state["question"],
                    "rows": "\n".join(state["rows"][:5]),
                    "rag": state["rag"],
                    "web": state["web"]
                }
                improved = (fusion_prompt | llm_retry).invoke(retry_input).content.strip()
                return {
                    "score": score,
                    "answer": improved + f"\n\n[🔁 Refined Answer due to low score]\nReason: {reason.strip()}"
                }
        except:
            pass
        return {"score": score, "answer": state["answer"]}

    builder = StateGraph(GraphState)
    builder.add_node("transform", query_transform_node)
    builder.add_node("csv", csv_row_retrieve_node)
    builder.add_node("rag", semantic_rag_node)
    builder.add_node("web", web_search_node)
    builder.add_node("synthesizer", summarize_fusion_node)
    builder.add_node("explain", explain_node)
    builder.add_node("score", score_node)

    builder.set_entry_point("transform")
    builder.add_edge("transform", "csv")
    builder.add_edge("transform", "rag")
    builder.add_edge("transform", "web")
    builder.add_edge("csv", "synthesizer")
    builder.add_edge("rag", "synthesizer")
    builder.add_edge("web", "synthesizer")
    builder.add_edge("synthesizer", "explain")
    builder.add_edge("explain", "score")
    builder.add_edge("score", END)

    return builder.compile()

# ✅ Streamlit UI
st.set_page_config(page_title="Real Estate Assistant (CSV Upload)", layout="wide")
st.title("🏡 AI Real Estate Assistant")

uploaded_file = st.file_uploader("📁 Upload a CSV file with apartment listings", type="csv")

if uploaded_file:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        retriever = setup_vectordb_from_df(df_uploaded)
        graph = build_graph(retriever)

        query = st.chat_input("Ask me about apartments...")
        if query:
            st.chat_message("user").write(query)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    result = graph.invoke({"question": query})
                    st.markdown("### 💬 Answer")
                    st.write(result["answer"])
                    st.markdown("### 🧠 Explanation")
                    st.write(result["explanation"])
                    st.markdown("### 🎯 Score")
                    st.success(result["score"])
    except Exception as e:
        st.error("❌ Error processing uploaded file.")
        st.exception(e)
