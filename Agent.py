from typing import List, Dict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from datetime import datetime


GEMINI_API_KEY = "AIzaSyCRM2WSP4-uX8mb0w0YktajHV1vPScVkcA"
TAVILY_API_KEY = "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"


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)


class NewsState(Dict):
    topic: str
    search_results: str
    summary: str

def search_tavily(state: NewsState) -> NewsState:
    topic = state["topic"]
    topic_News = topic + " news"
    print(f"Searching for news about '{topic}'...")
    results = tavily.search(topic_News)

    if not results.get("results"):
        print("Hmm, couldn't find anything useful.")
        state["search_results"] = "No relevant results found."
        return state

   
    snippets = []
    for r in results["results"][:4]:
        snippets.append(f"Title: {r['title']}\nSummary: {r['content']}")
    state["search_results"] = "\n\n".join(snippets)
    return state

def summarize_with_gemini(state: NewsState) -> NewsState:
    prompt = f"""
Hey! You're a journalist. Write a short summary  about '{state['topic']}' based ONLY on the info below.

Sources:
{state['search_results']}
"""
    response = llm.invoke(prompt)
    state["summary"] = response.content.strip()
    return state

def collect_digest(state: NewsState) -> NewsState:
    print(f"Here's a summary about '{state['topic']}':")
    print(state["summary"])
    print("-" * 40)
    return state

def build_graph():
    graph = StateGraph(NewsState)
    graph.add_node("search_tavily",  search_tavily)
    graph.add_node("summarize",    summarize_with_gemini)
    graph.add_node("collect",   collect_digest)

    graph.set_entry_point("search_tavily")
    graph.add_edge("search_tavily", "summarize")
    graph.add_edge("summarize", "collect")
    graph.add_edge("collect", END)

    return graph.compile()

def run_graph(topics: List[str]):
    print("Starting the news digest process...\n")
    graph=   build_graph()
    all_summaries= []

    for topic in topics:
        print(f"Working on topic: {topic}")
        state = {"topic": topic}
        final_state =graph.invoke(state)
        all_summaries.append(f"📰 {topic}:\n{final_state.get('summary', 'No summary')}\n")

    today = datetime.now().strftime("%Y-%m-%d")
    digest =f"*** Daily News Digest - {today} ***\n\n" + "\n".join(all_summaries)
    print("\n" +digest)
    return digest

if __name__ == "__main__":
    print("Hey! Enter the topics you want news on, separated by commas.")
    print("Example: tech, sports, politics, science")
    user_input= input("Your topics: ")

    topics =[t.strip() for t in user_input.split(",") if t.strip()]
    if not topics:
        print("Oops! You didn't enter any topics.")
    else:
        run_graph(topics)
