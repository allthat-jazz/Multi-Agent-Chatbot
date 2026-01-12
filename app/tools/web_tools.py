from langchain_tavily import TavilySearch

def build_web_tools(max_results=5):
    return [TavilySearch(max_results=int(max_results))]
