import os
import re
import streamlit as st
from datetime import datetime
from typing import TypedDict
from PyPDF2 import PdfReader
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

# 🔐 API Keys
os.environ["TAVILY_API_KEY"] = "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"
os.environ["TOGETHER_API_KEY"] = "fa8e3ca9951bfafbd565a680b69625467935ee44b94e518e9eb6fbd706d0d937"


# 🧠 LLM Setup
llm_main = ChatOpenAI(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    openai_api_base="https://api.together.xyz/v1",
    openai_api_key=os.environ["TOGETHER_API_KEY"],
    temperature=0.5
)
llm_score = ChatOpenAI(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    openai_api_base="https://api.together.xyz/v1",
    openai_api_key=os.environ["TOGETHER_API_KEY"],
    temperature=0.2
)
llm_retry = ChatOpenAI(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    openai_api_base="https://api.together.xyz/v1",
    openai_api_key=os.environ["TOGETHER_API_KEY"],
    temperature=0.9
)

# 🧠 Memory
memory = ConversationBufferMemory(return_messages=True)

# 🗃️ MongoDB Setup
client = MongoClient("mongodb://localhost:27017")
db = client["sap_testcase_gen"]
collection = db["session_logs"]
chunk_collection = db["doc_chunks"]

# 📄 Store chunks in Mongo
def store_chunks_in_mongo(file, file_type):
    reader = PdfReader(file)
    text = "\n".join([p.extract_text() or "" for p in reader.pages])
    chunks = text.split("\n\n")
    for chunk in chunks:
        cleaned = chunk.strip()
        if cleaned:
            chunk_collection.insert_one({"content": cleaned, "type": file_type})

# 🔎 Mongo Retriever
def mongo_retriever(query, file_type):
    all_docs = list(chunk_collection.find({"type": file_type}))
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = [Document(page_content=d["content"]) for d in all_docs]
    vectordb = DocArrayInMemorySearch.from_documents(docs, embed_model)
    return vectordb.as_retriever()

# 🔁 Graph State
class GraphState(TypedDict):
    question: str
    context: str
    web: str
    answer: str
    explanation: str
    score: str

# 🧠 Graph Construction
def build_graph():
    @RunnableLambda
    def retrieve_node(state: GraphState):
        file_type = "tds" if "tds" in state["question"].lower() else "fds"
        retriever = mongo_retriever(state["question"], file_type)
        docs = retriever.invoke(state["question"])
        return {"context": "\n".join([d.page_content for d in docs])}

    @RunnableLambda
    def web_search_node(state: GraphState):
        search_tool = TavilySearch(api_key=os.environ["TAVILY_API_KEY"], max_results=3)
        result = search_tool.invoke(state["question"])
        return {"web": result}

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a Test Case Generator for SAP. Generate edge-to-edge test cases from FDS/TDS. Format: Given-When-Then. Avoid hallucinations."),
        ("human", "User Message: {question}\nContext: {context}\nWeb: {web}")
    ])

    @RunnableLambda
    def answer_node(state: GraphState):
        memory.save_context({"input": state["question"]}, {"output": state.get("answer", "")})
        return {"answer": (answer_prompt | llm_main).invoke(state).content.strip()}

    explanation_prompt = ChatPromptTemplate.from_messages([
        ("system", "Explain why these test cases were generated. Link to FDS/TDS logic."),
        ("human", "Answer: {answer}\n\nContext: {context}\n\nWeb: {web}")
    ])

    @RunnableLambda
    def explain_node(state: GraphState):
        return {"explanation": (explanation_prompt | llm_main).invoke(state).content.strip()}

    score_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a QA Agent. Score from 1-10. If <7, suggest and regenerate."),
        ("human", "Question: {question}\nAnswer: {answer}\nExplanation: {explanation}")
    ])

    @RunnableLambda
    def score_node(state: GraphState):
        result = (score_prompt | llm_score).invoke(state).content
        match = re.search(r"\b(\d{1,2})\b", result)
        score = match.group(1) if match else "7"
        if int(score) < 7:
            improved = (answer_prompt | llm_retry).invoke(state).content.strip()
            return {"score": score, "answer": improved + "\n\n[Improved after feedback loop]"}
        return {"score": score}

    # Build Graph
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

# 🌐 Streamlit UI
st.set_page_config("SAP Test Case Generator")
st.title("🧪 SAP Test Case Generator (MAS, Mongo, Feedback)")

file = st.file_uploader("📄 Upload FDS or TDS PDF", type="pdf")

if file:
    try:
        file_type = "tds" if "tds" in file.name.lower() else "fds"
        store_chunks_in_mongo(file, file_type)
        graph = build_graph()

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        query = st.chat_input("Ask to generate test cases for scenario...")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            with st.chat_message("assistant"):
                with st.spinner("Generating test cases..."):
                    result = graph.invoke({"question": query})
                    log_data = {"timestamp": datetime.utcnow(), **result, "question": query}
                    collection.insert_one(log_data)

                    output = f"### ✅ Test Cases\n{result['answer']}\n\n---\n### 💡 Explanation\n{result['explanation']}\n\n🎯 Score: {result['score']}"
                    st.session_state.chat_history.append({"role": "assistant", "content": output})
                    st.markdown(output)

    except Exception as e:
        st.error("❌ Upload failed or processing error")
        st.exception(e)
