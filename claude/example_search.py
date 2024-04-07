# %% Setup
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever

retriever = TavilySearchAPIRetriever(k=3)

message = retriever.invoke("what year was breath of the wild released?")
print(message)
