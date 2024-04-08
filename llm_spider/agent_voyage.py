import argparse
import asyncio
import base64
import io
import os
import platform
import re
from getpass import getpass
from typing import List, Optional, TypedDict

import nest_asyncio
from attr import dataclass
from dotenv import load_dotenv
from langchain_core import prompts
from langchain_core.messages import BaseMessage, SystemMessage, ai, chat, function, human, system, tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables import chain as chain_decorator
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from PIL import Image
from playwright.async_api import Page, async_playwright
from playwright_stealth import stealth_async

from path import Path


def _getpass(env_var: str):
    if not os.environ.get(env_var):
        os.environ[env_var] = getpass(f"{env_var}=")


load_dotenv()
_getpass("OPENAI_API_KEY")

# So overview, we have:

# 1. Graph State
# 2. Tools
# 3. Agent
# 4. Graph
# 5. Run agent

# ----------------------------

nest_asyncio.apply()

# 1. Graph State:


# Bounding box
class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str


class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]


class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # The bounding boxes from the browser annotation function
    prediction: Prediction  # The Agent's output
    # A system message (or messages) containing the intermediate steps
    scratchpad: List[BaseMessage]
    observation: str  # The most recent response from a tool


# 2. Tools:


async def click(state: AgentState):
    # - Click [Numerical_Label]
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = click_args[0]
    bbox_id = int(bbox_id)

    try:
        bbox = state["bboxes"][bbox_id]
    except IndexError:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)

    return f"Clicked {bbox_id}"


async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return f"Failed to type in element from bounding box labeled as number {type_args}"
    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    # Check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"


async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
        # Not sure the best value for this:
        scroll_amount = 500
        scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        # Scrolling within a specific element
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"


async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."


async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."


async def to_google(state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."


# 3. Agent:

# 1. 現在のページにバウンディングボックスを注釈する `mark_page` 関数
# 2. ユーザーの質問、注釈付きの画像、Agentのスクラッチパッドを保持するプロンプト
# 3. 次のステップを決定するGPT-4V
# 4. アクションを抽出するパーシングロジック

with open("mark_page.js") as f:
    mark_page_script = f.read()


@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    bboxes = []
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            await asyncio.sleep(3)
    screenshot = await page.screenshot()
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }


# Agent prompt


async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}


def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}


def parse(text: str) -> dict:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [inp.strip().strip("[]") for inp in action_input.strip().split(";")]
    return {"action": action, "args": action_input}


# prompt = hub.pull("wfh/web-voyager")
prompt = prompts.ChatPromptTemplate(
    input_variables=["bbox_descriptions", "img", "input"],
    input_types={
        "scratchpad": list[
            ai.AIMessage
            | human.HumanMessage
            | chat.ChatMessage
            | system.SystemMessage
            | function.FunctionMessage
            | tool.ToolMessage
        ]
    },
    partial_variables={"scratchpad": []},
    messages=[
        prompts.SystemMessagePromptTemplate(
            prompt=[
                prompts.PromptTemplate(
                    input_variables=[],
                    template="""Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot will
feature Numerical Labels placed in the TOP LEFT corner of each Web Element. Carefully analyze the visual
information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow
the guidelines and choose one of the following actions:

1. Click a Web Element.
2. Delete existing content in a textbox and then type content.
3. Scroll up or down.
4. Wait 
5. Go back
7. Return to google to start over.
8. Respond with the final answer

Correspondingly, Action should STRICTLY follow the format:

- Click [Numerical_Label] 
- Type [Numerical_Label]; [Content] 
- Scroll [Numerical_Label or WINDOW]; [up or down] 
- Wait 
- GoBack
- Google
- ANSWER; [content]

Key Guidelines You MUST follow:

* Action guidelines *
1) Execute only one action per iteration.
2) When clicking or typing, ensure to select the correct bounding box.
3) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.

* Web Browsing Guidelines *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages
2) Select strategically to minimize time wasted.

Your reply should strictly follow the format:

Thought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}
Action: {{One Action format you choose}}
Then the User will provide:
Observation: {{A labeled screenshot Given by User}}
""",
                )
            ]
        ),
        prompts.MessagesPlaceholder(variable_name="scratchpad", optional=True),
        prompts.HumanMessagePromptTemplate(
            prompt=[
                ImagePromptTemplate(input_variables=["img"], template={"url": "data:image/png;base64,{img}"}),
                prompts.PromptTemplate(input_variables=["bbox_descriptions"], template="{bbox_descriptions}"),
                prompts.PromptTemplate(input_variables=["input"], template="{input}"),
            ]
        ),
    ],
)

llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=4096)
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

# 4. Graph:


def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad")
    if old and isinstance(old[0], str):
        txt = str(old[0])
        last_line = txt.split("\n")[-1]
        ma = re.match(r"\d+", last_line)
        if ma is None:
            txt = "Previous action observations:\n"
            step = 1
        else:
            step = int(ma.group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"

    return {**state, "scratchpad": [SystemMessage(content=txt)]}


graph_builder = StateGraph(AgentState)

# Nodes (doing the work)
graph_builder.add_node("agent", agent)
graph_builder.set_entry_point("agent")

# Edges (data flow)
graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
}


for node_name in tools:
    graph_builder.add_node(
        node_name,
        RunnableLambda(tools[node_name]) | (lambda observation: {"observation": observation}),
    )
    graph_builder.add_edge(node_name, "update_scratchpad")


def select_tool(state: AgentState):
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action


graph_builder.add_conditional_edges("agent", select_tool)
graph = graph_builder.compile()

# 5. Run agent


async def call_agent(question: str, page, max_steps: int = 150):
    path = Path()
    objective_image = path.create_text_image("Objective: " + question, width=800, height=100, font_size=100)
    path.update_agent_path_image(objective_image, is_initial=True)

    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )

    final_answer = None
    steps = []
    step_counter = 0
    async for event in event_stream:
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")

        step_counter += 1
        steps.append(f"{step_counter}. {action}: {action_input}")

        with open("agent_steps.txt", "w") as file:
            file.write("\n".join(steps))

        screenshot_data = base64.b64decode(event["agent"]["img"])
        img = Image.open(io.BytesIO(screenshot_data))
        path.update_agent_path_image(img)

        if action and "ANSWER" in action:
            if action_input is None:
                raise ValueError("No answer provided.")
            final_answer = action_input[0]
            # Create and add the final response image
            final_response_image = path.create_text_image(
                "Final Response: " + final_answer, width=800, height=100, font_size=20
            )
            path.update_agent_path_image(final_response_image, is_final=True)
            break

    return final_answer


# Main


@dataclass
class Args:
    objective: str


async def main():
    parser = argparse.ArgumentParser(description="Run the agent on a given objective")
    parser.add_argument("--objective", type=str, help="The question to run the agent on")
    args = Args(**vars(parser.parse_args()))

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await stealth_async(page)
        await page.goto("https://www.google.com")

        try:
            res = await call_agent(args.objective, page)
            print(f"Final response: {res}")
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
