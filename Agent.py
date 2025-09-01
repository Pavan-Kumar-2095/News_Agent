from typing import List, Dict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from datetime import datetime
import faiss
import threading
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

GEMINI_API_KEY = ""
TAVILY_API_KEY = ""
RAG_FILE = "rag_knowledge.jsonl"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)


faiss_index = None
faiss_queries = []
faiss_summaries = []
faiss_lock = threading.Lock()


class NewsState(Dict):
    topic: str
    search_results: str
    summary: str


def embed_text(text: str) -> np.ndarray:
    vec = embedding_model.encode([text], convert_to_numpy=True)[0]
    return vec / np.linalg.norm(vec)


def save_to_knowledge_file(query: str, summary: str, filename=RAG_FILE):
    data = {
        "query": query,
        "summary": summary,
        "timestamp": datetime.now().isoformat()
    }
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")

def load_faiss_from_file(filename=RAG_FILE):
    global faiss_index, faiss_queries, faiss_summaries
    if not os.path.exists(filename):
        return

    vectors, queries, summaries = [], [], []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                query = record["query"]
                summary = record["summary"]
                vec = embed_text(query).astype("float32")
                vectors.append(vec)
                queries.append(query)
                summaries.append(summary)
            except Exception as e:
                print(f"[!] Failed to load record: {e}")

    if vectors:
        dim = len(vectors[0])
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(np.array(vectors))
        faiss_queries = queries
        faiss_summaries = summaries
        print(f"[✓] Loaded {len(vectors)} cached summaries from {filename}")


def add_summary_to_faiss(query: str, summary: str):
    global faiss_index, faiss_queries, faiss_summaries
    vec = embed_text(query).astype('float32')

    with faiss_lock:
        if faiss_index is None:
            faiss_index = faiss.IndexFlatIP(len(vec))
        faiss_index.add(np.array([vec]))
        faiss_queries.append(query.strip().lower())
        faiss_summaries.append(summary)

    save_to_knowledge_file(query, summary)
    print(f"[✓] Cached summary for '{query}' in FAISS and saved to RAG file.")

def search_faiss_for_summary(query: str, threshold=0.75) -> str | None:
    global faiss_index, faiss_queries, faiss_summaries
    query = query.strip().lower()

    print(f"[FAISS] Searching cache for: '{query}'")
    if faiss_index is None or not faiss_queries:
        print("[FAISS] Cache is empty.")
        return None

    vec = embed_text(query).astype('float32').reshape(1, -1)

    with faiss_lock:
        distances, indices = faiss_index.search(vec, 1)

    score, idx = distances[0][0], indices[0][0]
    print(f"[FAISS] Similarity score: {score:.4f}")

    if score >= threshold and idx < len(faiss_summaries):
        print(f"[✓] Found cached result for '{query}'")
        return faiss_summaries[idx]

    print("[FAISS] No good match found.")
    return None


def search_tavily(state: NewsState) -> NewsState:
    topic = state["topic"]
    query = topic + " news"
    print(f"🔍 Searching Tavily for '{topic}'...")
    results = tavily.search(query)

    if not results.get("results"):
        state["search_results"] = "No relevant results found."
        return state

    snippets = [f"Title: {r['title']}\nSummary: {r['content']}" for r in results["results"][:4]]
    state["search_results"] = "\n\n".join(snippets)
    return state

def summarize_with_gemini(state: NewsState) -> NewsState:
    print("✏️ Getting summary from Gemini...")
    prompt = f"""
You are a journalist. Write a short and clear summary about the topic: '{state['topic']}'.
Use ONLY the information provided below. Be factual, concise, and neutral.

Sources:
{state['search_results']}
"""
    response = llm.invoke(prompt)
    state["summary"] = response.content.strip()
    return state

def collect_digest(state: NewsState) -> NewsState:
    print(f"\n📢 Summary for '{state['topic']}':")
    print(state["summary"])
    print("-" * 50)
    return state

def build_search_graph():
    graph = StateGraph(NewsState)
    graph.add_node("search_tavily", search_tavily)
    graph.add_node("summarize", summarize_with_gemini)
    graph.add_node("collect", collect_digest)

    graph.set_entry_point("search_tavily")
    graph.add_edge("search_tavily", "summarize")
    graph.add_edge("summarize", "collect")
    graph.add_edge("collect", END)

    return graph.compile()


def get_summary_for_topic(topic: str) -> str:
    topic = topic.strip()
    cache_key = topic.lower() + " news"
    state = {"topic": topic, "search_results": "", "summary": ""}

    cached = search_faiss_for_summary(cache_key)
    if cached:
        print(f"[CACHE] Using cached summary for '{topic}'")
        state["summary"] = cached
        collect_digest(state)
        return cached

    print(f"[CACHE MISS] Running Tavily + Gemini for '{topic}'")
    graph = build_search_graph()
    final_state = graph.invoke(state)

    summary = final_state.get("summary", "No summary available.")
    add_summary_to_faiss(cache_key, summary)
    return summary

def run_graph(topics: List[str]):
    print("\n🌐 Starting Daily News Digest...\n")
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"*** Daily News Digest - {today} ***\n")

    all_summaries = []

    for topic in topics:
        print(f"\n➡️ Processing topic: {topic}")
        summary = get_summary_for_topic(topic)
        all_summaries.append(f"📰 {topic}:\n{summary}\n")
        print("=" * 50)

    print("\n✅ Final Summary:\n")
    print(f"*** Daily News Digest - {today} ***\n")
    print("\n".join(all_summaries))


if __name__ == "__main__":
    print("🔁 Loading RAG knowledge base...")
    load_faiss_from_file()

    print("🔎 Enter the topics you want news on, separated by commas.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Your topics: ").strip()
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        topics = [t.strip() for t in user_input.split(",") if t.strip()]
        if not topics:
            print("⚠️ No topics entered. Try again.")
            continue

        run_graph(topics)
