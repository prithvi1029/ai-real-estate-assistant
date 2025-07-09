import os
import re
import traceback
import streamlit as st
import pandas as pd
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
from pymongo import MongoClient

# Mongo helper setup
def get_collection(collection_name):
    client = MongoClient("mongodb://localhost:27017")
    db = client["real_estate"]
    return db[collection_name]

# 🔐 API Keys
os.environ["TAVILY_API_KEY"] = "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"
os.environ["TOGETHER_API_KEY"] = "fa8e3ca9951bfafbd565a680b69625467935ee44b94e518e9eb6fbd706d0d937"


# 🧠 LLMs
llm_transform = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.7)
llm_main = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.5)
llm_retry = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.9)
llm_score = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.3)

# ✅ Graph State
class GraphState(TypedDict):
    question: str
    transformed_question: str
    rows: list
    rag: str
    web: str
    mongo: str
    answer: str
    explanation: str
    score: str

def preprocess_apartment_data(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = ["type", "ownership", "buildingMaterial", "condition", "hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"]
    for col in categorical_columns:
        df[col] = df[col].fillna("unknown").astype(str).str.lower()

    distance_columns = [col for col in df.columns if "Distance" in col or col in ["squareMeters", "price", "centreDistance"]]
    df[distance_columns] = df[distance_columns].round(2)

    df["floorCategory"] = pd.cut(df["floor"].fillna(-1), bins=[-2, 0, 3, 6, 100], labels=["basement", "low", "mid", "high"])
    df["floorCountCategory"] = pd.cut(df["floorCount"].fillna(-1), bins=[-2, 3, 6, 10, 100], labels=["small", "medium", "large", "highrise"])
    current_year = datetime.utcnow().year
    df["buildingAge"] = current_year - df["buildYear"].fillna(current_year)
    df["buildingAgeCategory"] = pd.cut(df["buildingAge"], bins=[-1, 5, 15, 30, 1000], labels=["new", "modern", "old", "very old"])
    for col in ["hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"]:
        df[col] = df[col].str.strip().str.lower().replace({"1": "yes", "0": "no"})

    return df.dropna(subset=["price", "rooms", "squareMeters", "city"])

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

    @RunnableLambda
    def web_search_node(state: GraphState):
        search_tool = TavilySearch(api_key=os.environ["TAVILY_API_KEY"], max_results=3)
        content = search_tool.invoke(state["transformed_question"] + " in Poland")
        return {"web": content}

    @RunnableLambda
    def mongo_query_node(state: GraphState):
        try:
            client = MongoClient("mongodb://localhost:27017")
            db = client["real_estate"]
            logs = db["session_logs"]
            past = logs.find_one(sort=[("timestamp", -1)])
            return {"mongo": str(past) if past else "No recent logs found."}
        except Exception as e:
            return {"mongo": f"Mongo query failed: {str(e)}"}

    fusion_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a helpful real estate assistant using CSV and RAG. Prioritize listings."),
        ("human", "Question: {question}\nCSV Rows:\n{rows}\nRAG:\n{rag}\nWeb:\n{web}\nMongo Logs:\n{mongo}")
    ])

    @RunnableLambda
    def summarize_fusion_node(state: GraphState):
        result = (fusion_prompt | llm_main).invoke({
            "question": state["question"],
            "rows": "\n".join(state["rows"][:5]),
            "rag": state["rag"],
            "web": state["web"],
            "mongo": state["mongo"]
        })
        return {"answer": result.content.strip()}

    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a reasoning analyst. Explain how the answer was derived."),
        ("human", "RAG: {rag}\n\nRows: {rows}\n\nWeb: {web}")
    ])

    @RunnableLambda
    def explain_node(state: GraphState):
        return {
            "explanation": llm_main.invoke(explain_prompt.format(
                rag=state["rag"], rows="\n".join(state["rows"]), web=state["web"]
            )).content.strip()
        }

    score_prompt = ChatPromptTemplate.from_messages([
        ("system", "Score from 1-10 how well the answer fits the user's query."),
        ("human", "Query: {question}\n\nAnswer: {answer}\n\nExplanation: {explanation}")
    ])

    @RunnableLambda
    def score_node(state: GraphState):
        res = llm_score.invoke(score_prompt.format(
            question=state["question"], answer=state["answer"], explanation=state["explanation"]
        )).content.strip()

        match = re.search(r"\b(\d{1,2})\b", res)
        score = match.group(1) if match else "7"
        reason = res

        try:
            if int(score) < 7:
                improved = (fusion_prompt | llm_retry).invoke({
                    "question": state["question"],
                    "rows": "\n".join(state["rows"][:5]),
                    "rag": state["rag"],
                    "web": state["web"],
                    "mongo": state["mongo"]
                }).content.strip()
                return {"score": score, "answer": improved + f"\n\n[🔁 Refined Answer due to low score]\nReason: {reason.strip()}"}
        except Exception:
            return {"score": "7", "answer": state["answer"] + "\n\n[⚠️ Score parsing failed, fallback applied]"}

        return {"score": score, "answer": state["answer"]}

    builder = StateGraph(GraphState)
    builder.add_node("transform", query_transform_node)
    builder.add_node("csv", csv_row_retrieve_node)
    builder.add_node("rag", semantic_rag_node)
    builder.add_node("web", web_search_node)
    builder.add_node("mongo", mongo_query_node)
    builder.add_node("synthesizer", summarize_fusion_node)
    builder.add_node("explain", explain_node)
    builder.add_node("score", score_node)
    builder.set_entry_point("transform")
    builder.add_edge("transform", "csv")
    builder.add_edge("transform", "rag")
    builder.add_edge("transform", "web")
    builder.add_edge("transform", "mongo")
    builder.add_edge("csv", "synthesizer")
    builder.add_edge("rag", "synthesizer")
    builder.add_edge("web", "synthesizer")
    builder.add_edge("mongo", "synthesizer")
    builder.add_edge("synthesizer", "explain")
    builder.add_edge("explain", "score")
    builder.add_edge("score", END)
    return builder.compile()

# ✅ Streamlit Chatbot UI
st.set_page_config(page_title="AI Real Estate Chatbot", layout="wide")
st.title("🏡 AI Real Estate Assistant (Conversational)")

log_collection = get_collection("session_logs")
uploaded_file = st.file_uploader("📁 Upload apartment CSV", type="csv")

if uploaded_file:
    try:
        df_uploaded = preprocess_apartment_data(pd.read_csv(uploaded_file))
        retriever = setup_vectordb_from_df(df_uploaded)
        graph = build_graph(retriever)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        query = st.chat_input("Ask me about apartments...")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    result = graph.invoke({"question": query})

                    log_data = {
                        "timestamp": datetime.utcnow(),
                        "question": query,
                        "transformed_question": result.get("transformed_question", ""),
                        "rows": result.get("rows", []),
                        "rag": result.get("rag", ""),
                        "web": result.get("web", ""),
                        "mongo": result.get("mongo", ""),
                        "answer": result.get("answer", ""),
                        "explanation": result.get("explanation", ""),
                        "score": result.get("score", "")
                    }
                    log_collection.insert_one(log_data)

                    response = f"### 💬 Answer\n{result['answer']}\n\n---\n### 🧠 Explanation\n{result['explanation']}\n\n### 🎯 Score: {result['score']}"
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.markdown(response)

    except Exception as e:
        st.error("❌ Error processing uploaded file.")
        st.exception(e)