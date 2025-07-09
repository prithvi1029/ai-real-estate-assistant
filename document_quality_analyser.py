import os
import re
import streamlit as st
from docx import Document as DocxDoc
from typing import List, Dict, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from pymongo import MongoClient
from datetime import datetime

# --- MongoDB Logging ---
def get_collection(name):
    client = MongoClient("mongodb://localhost:27017")
    db = client["doc_quality"]
    return db[name]

# --- API Keys ---
# 🔐 API Keys
os.environ["TAVILY_API_KEY"] = "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"
os.environ["TOGETHER_API_KEY"] = "fa8e3ca9951bfafbd565a680b69625467935ee44b94e518e9eb6fbd706d0d937"

# --- LLMs ---
llm_compare = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.4)
llm_score = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.2)
llm_retry = ChatOpenAI(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"], temperature=0.8)

# --- Document Processing ---
def extract_sections(doc: DocxDoc) -> Dict[str, str]:
    section_map = {}
    current_heading = "Introduction"
    current_content = []
    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            if current_content:
                section_map[current_heading] = "\n".join(current_content)
                current_content = []
            current_heading = para.text.strip()
        else:
            current_content.append(para.text.strip())
    if current_content:
        section_map[current_heading] = "\n".join(current_content)
    return section_map

# --- Agent State ---
class GraphState(TypedDict):
    section: str
    reference: str
    candidate: str
    feedback: str
    score: str
    retry_output: str

# --- Prompt Templates ---
compare_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document comparison agent. Compare section-wise quality of technical documents."),
    ("human", "Section: {section}\n\nReference:\n{reference}\n\nCandidate:\n{candidate}\n\nProvide feedback on completeness, clarity, and correctness.")
])

score_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a scoring agent for document quality."),
    ("human", "Section: {section}\n\nFeedback:\n{feedback}\n\nGive a score out of 10 and explain why.")
])

retry_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a corrective agent. Rewrite or suggest improvements for the candidate based on feedback."),
    ("human", "Section: {section}\n\nCandidate:\n{candidate}\n\nFeedback:\n{feedback}")
])

# --- Graph Build ---
def build_graph():
    @RunnableLambda
    def compare_node(state: GraphState):
        response = (compare_prompt | llm_compare).invoke(state).content.strip()
        return {"feedback": response}

    @RunnableLambda
    def score_node(state: GraphState):
        response = (score_prompt | llm_score).invoke(state).content.strip()
        score_match = re.search(r"\b(\d{1,2})\b", response)
        score = score_match.group(1) if score_match else "7"
        return {"score": score, "feedback": state["feedback"] + f"\n\nScoring: {response}"}

    @RunnableLambda
    def retry_node(state: GraphState):
        if int(state["score"]) < 7:
            retry_text = (retry_prompt | llm_retry).invoke(state).content.strip()
            return {"retry_output": retry_text}
        return {"retry_output": "[No retry needed]"}

    builder = StateGraph(GraphState)
    builder.add_node("compare", compare_node)
    builder.add_node("score", score_node)
    builder.add_node("retry", retry_node)

    builder.set_entry_point("compare")
    builder.add_edge("compare", "score")
    builder.add_edge("score", "retry")
    builder.add_edge("retry", END)

    return builder.compile()

# --- Streamlit UI ---
st.set_page_config("📄 Document Quality Analyzer")
st.title("📑 Document Quality Analyzer Agent (FDS/TDS)")

ref_doc = st.file_uploader("📘 Upload Reference Document", type=["docx"])
cand_doc = st.file_uploader("📙 Upload Candidate Document", type=["docx"])
collection = get_collection("section_logs")

if ref_doc and cand_doc:
    ref_sections = extract_sections(DocxDoc(ref_doc))
    cand_sections = extract_sections(DocxDoc(cand_doc))

    common_sections = list(set(ref_sections) & set(cand_sections))
    if not common_sections:
        st.warning("No matching sections found.")
    else:
        graph = build_graph()
        for section in common_sections:
            st.subheader(f"📌 Section: {section}")
            result = graph.invoke({
                "section": section,
                "reference": ref_sections[section],
                "candidate": cand_sections[section]
            })
            collection.insert_one({
                "section": section,
                "reference": ref_sections[section],
                "candidate": cand_sections[section],
                "timestamp": datetime.utcnow(),
                **result
            })
            st.markdown(f"### 📝 Feedback\n{result['feedback']}")
            st.markdown(f"### 📊 Score: {result['score']}")
            if result['retry_output'] and "[No retry needed]" not in result['retry_output']:
                st.markdown(f"### 🔁 Suggested Rewrite\n{result['retry_output']}")
