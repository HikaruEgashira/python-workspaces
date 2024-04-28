import argparse
import warnings

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt.chat_agent_executor import create_tool_calling_executor
from langgraph.prebuilt.tool_node import ToolNode

warnings.filterwarnings("ignore")


# 条件付きエッジ
def should_continue(messages: list[AIMessage]):
    last_message = messages[-1]
    if last_message.tool_calls:
        return "action"
    return END


# 利用するツールとモデルを定義
tools = [TavilySearchResults(max_results=1)]
# model = ChatAnthropic(model_name="claude-3-haiku-20240307")
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ワークフローを定義
workflow = MessageGraph()
workflow.add_node("agent", model.bind_tools(tools))
workflow.add_node("action", ToolNode(tools))
workflow.add_edge("action", "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.set_entry_point("agent")

memory = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=memory)
# app = create_tool_calling_executor(model), tools)


# CLIを定義
parser = argparse.ArgumentParser(description="Run a language graph")
parser.add_argument("-p", "--print", action="store_true", help="Print the graph")
parser.add_argument("question", type=str, help="The question to ask the graph", nargs="?")
args = parser.parse_args()
if args.print:
    print(app.get_graph().draw_mermaid())
    exit(0)

# 実行
question = args.question or "2024/01/01の東京の天気は？"
thread = {"configurable": {"thread_id": "4"}}
for step in app.stream(question, thread):  # type: ignore
    node, message = next(iter(step.items()))
    if message:
        if isinstance(message, list) and isinstance(message[-1], BaseMessage):
            message[-1].pretty_print()
        elif isinstance(message, BaseMessage):
            message.pretty_print()
            if isinstance(message, AIMessage) and len(message.tool_calls) > 0:
                print(message.tool_calls)
