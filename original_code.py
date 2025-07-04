import os
import pandas as pd
from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch

# 🔐 API Keys
os.environ["TAVILY_API_KEY"] = "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"
os.environ["TOGETHER_API_KEY"] = "fa8e3ca9951bfafbd565a680b69625467935ee44b94e518e9eb6fbd706d0d937"

# ✅ Load Data
csv_url = "D:/real_estate_assistant_dataset/pl/apartments_rent_pl_2024_01.csv"
df = pd.read_csv(csv_url).dropna()
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Vector Store
def enrich_row(row):
    return f"{row['rooms']} room apartment in {row['city']} priced at {row['price']} PLN. Area: {row['squareMeters']} sqm. Parking: {row['hasParkingSpace']}. Balcony: {row['hasBalcony']}. Elevator: {row['hasElevator']}. Condition: {row['condition']}."

docs = [Document(page_content=enrich_row(row)) for _, row in df.iterrows()]
vstore = DocArrayInMemorySearch.from_documents(docs, embed_model)
retriever = vstore.as_retriever(search_type="mmr")

# ✅ LLM Setup
llm_transform = ChatOpenAI(temperature=0.7, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"])
llm_main = ChatOpenAI(temperature=0.5, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"])
llm_retry = ChatOpenAI(temperature=0.9, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"])
llm_score = ChatOpenAI(temperature=0.3, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", openai_api_base="https://api.together.xyz/v1", openai_api_key=os.environ["TOGETHER_API_KEY"])

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

# ✅ Nodes
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

# ✅ FIXED Web Search Node
search_tool = TavilySearch(api_key=os.environ["TAVILY_API_KEY"], max_results=3)
@RunnableLambda
def web_search_node(state: GraphState):
    content = search_tool.invoke(state["transformed_question"] + " in Poland")
    return {"web": content}

fusion_prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful real estate assistant using CSV and RAG. Prioritize listings."),
    ("human", "Question: {question}\nCSV Rows:\n{rows}\nRAG:\n{rag}\nWeb:\n{web}\nAnswer:")
])
@RunnableLambda
def summarize_fusion_node(state: GraphState):
    rows_text = "\n".join(state.get("rows", [])[:5])
    result = llm_main.invoke(fusion_prompt.format(
        question=state["question"],
        rows=rows_text,
        rag=state["rag"],
        web=state["web"]
    ))
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
            improved = llm_retry.invoke(fusion_prompt.format(
                question=state["question"],
                rows="\n".join(state["rows"][:5]),
                rag=state["rag"],
                web=state["web"]
            )).content.strip()
            return {
                "score": score,
                "answer": improved + f"\n\n[🔁 Refined Answer due to low score]\nReason: {reason.strip()}"
            }
    except:
        pass
    return {"score": score, "answer": state["answer"]}

# ✅ Graph Definition
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

graph = builder.compile()

# ✅ Test Queries
test_queries = [
   # "Find me a 2-room apartment in Krakow under 3000 with parking.",
    "Find a 3-room apartment in Warsaw near the city center under 4000 PLN.",
   # "Are there any pet-friendly apartments in Gdansk with a balcony and underground parking?",
   # "Which cities in Poland have the cheapest apartments under 2500 PLN with 2 bedrooms?",
   # "Show listings for apartments in Wroclaw with a garden and at least 60 square meters.",
   # "Find studio apartments in Poznan below 2000 PLN with parking and good public transport.",
   # "List the top 3 best value-for-money apartments in Lublin for students under 1800 PLN."
]

# ✅ Execute
for q in test_queries:
    print("\n🟡 QUESTION:", q)
    res = graph.invoke({"question": q})
    print("✅ ANSWER:\n", res["answer"])
    print("🧠 EXPLANATION:\n", res["explanation"])
    print("🎯 SCORE:", res["score"])
