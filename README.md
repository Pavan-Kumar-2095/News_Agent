# News Digest Generator

This Python script creates a daily news digest based on user-specified topics. It searches for recent news articles using the Tavily API, summarizes the news using Google Gemini AI, and presents a concise digest.

---

## Features

- Use **langgraph** to define a modular AI agent workflow with multiple states
- Search for recent news articles on various topics via Tavily API
- Summarize news snippets using Google Gemini, a powerful generative AI model
- Produce a clean, concise daily news digest in a fully automated pipeline
- Easily extensible by adding or modifying states in the langgraph

---
## Example Run

Hey! Enter the topics you want news on, separated by commas.

Example: tech, sports, politics, science

Your topics: indian tech, indian sports 

Starting the news digest process...
----------------------------------------

*** Daily News Digest - 2025-08-27 ***

📰 indian tech:
Indian tech news currently focuses on new product launches. Recent releases include new mobile phones, laptops, gaming gadgets, and other electronics from various brands such as Sony and Samsung. Specific examples include Sony's ULT Power Sound series and Samsung's 2025 Soundbar lineup, along with a Milagrow robotic vacuum cleaner. Pricing details are often included in the reporting.

📰 indian sports:
Indian sports news is currently dominated by several key events and announcements. Ravichandran Ashwin's retirement from the Indian Premier League (IPL) is making headlines, alongside celebrations of India's T20 World Cup victory. The women's national hockey team is preparing for the Junior World Cup in 2025, and the Indian women's cricket team will host Australia in an ODI series before the World Cup. Additionally, India's senior bridge team secured a silver medal in the World Bridge Olympiad. News sources also cover a broader range of sports, including Pro Kabaddi League updates.


## Requirements

- Python 3.8+
- API keys for:
  - [Google Gemini](https://developers.google.com/) (Google Generative AI)
  - [Tavily](https://tavily.com/) (news search API)

- Install dependencies:
```bash
pip install langgraph langchain_google_genai tavily
