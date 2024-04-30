import argparse
import asyncio
import operator
from typing import Annotated, Sequence, TypedDict

import nest_asyncio
from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_node import ToolNode
from playwright.async_api import Page, async_playwright
from playwright_stealth import stealth_async

nest_asyncio.apply()
async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = [*toolkit.get_tools()]

model = ChatOpenAI(temperature=0, streaming=True)
model = model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    page: Page


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "action"
    else:
        return END


async def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response], "page": state["page"]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("action", "agent")
app = workflow.compile()


def p(message):
    if isinstance(message, AIMessage) and len(message.tool_calls) > 0:
        print(message.tool_calls)
    elif isinstance(message, BaseMessage):
        message.pretty_print()


async def main():
    parser = argparse.ArgumentParser(description="Run the agent on a given objective")
    parser.add_argument("--objective", type=str, help="The question to run the agent on")
    args = parser.parse_args()

    objective = args.objective or "What is the capital of France?"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await stealth_async(page)
        await page.goto("https://www.google.com")

        try:
            event_stream = app.astream(
                {
                    "page": page,
                    "messages": [HumanMessage(content=objective)],
                },
                {
                    "recursion_limit": 10,
                    "configurable": {"thread_id": "4"},
                },
            )
            async for event in event_stream:
                print(event)
        finally:
            await browser.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
