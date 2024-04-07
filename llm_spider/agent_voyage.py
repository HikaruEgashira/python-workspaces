import argparse
import asyncio
import base64
import os
import platform
import re
from getpass import getpass
from typing import List, Optional, TypedDict

import nest_asyncio
from langchain import hub
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables import chain as chain_decorator
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from PIL import Image
from playwright.async_api import Page, async_playwright
from playwright_stealth import stealth_async


def _getpass(env_var: str):
    if not os.environ.get(env_var):
        os.environ[env_var] = getpass(f"{env_var}=")


from dotenv import load_dotenv

load_dotenv()
_getpass("OPENAI_API_KEY")

# So overview, we have:

# 1. Define graph state
# 2. Define tools
# 3. Define agent
# 4. Define graph
# 5. Run agent

# ----------------------------

nest_asyncio.apply()

# 1. Define Graph State:
# 状態は、グラフ内の各ノードに入力を提供します。この場合、エージェントは、ウェブページオブジェクト（ブラウザ内）、注釈付き画像+バウンディングボックス、ユーザーの初期リクエスト、エージェントスクラッチパッド、システムプロンプトなどの情報を含むメッセージを追跡します。


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


# This represents the state of the agent
# as it proceeds through execution
class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # The bounding boxes from the browser annotation function
    prediction: Prediction  # The Agent's output
    # A system message (or messages) containing the intermediate steps
    scratchpad: List[BaseMessage]
    observation: str  # The most recent response from a tool


# 2. Define tools

# We define them below here as functions:


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
    except:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    res = await page.mouse.click(x, y)
    # TODO: In the paper, they automatically parse any downloaded PDFs
    # Could add something similar here as well and generally improve response format.
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


# 3a. Define Agent:

# Agentはマルチモーダルモデルによって駆動され、各ステップで取るべきアクションを決定します。それは以下のいくつかの実行可能なオブジェクトで構成されています:

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
        except:
            await asyncio.sleep(3)
    screenshot = await page.screenshot()
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }


# 3b. Agent definition:


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


# Will need a later version of langchain to pull this image prompt template
prompt = hub.pull("wfh/web-voyager")

# ChatPromptTemplate(input_variables=['bbox_descriptions', 'img', 'input'], input_types={'scratchpad': typing.List[typing.Union[langchain_core.messages.ai.AIMessage, langchain_core.messages.human.HumanMessage, langchain_core.messages.chat.ChatMessage, langchain_core.messages.system.SystemMessage, langchain_core.messages.function.FunctionMessage, langchain_core.messages.tool.ToolMessage]]}, partial_variables={'scratchpad': []}, messages=[SystemMessagePromptTemplate(prompt=[PromptTemplate(input_variables=[], template="Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot will\nfeature Numerical Labels placed in the TOP LEFT corner of each Web Element. Carefully analyze the visual\ninformation to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow\nthe guidelines and choose one of the following actions:\n\n1. Click a Web Element.\n2. Delete existing content in a textbox and then type content.\n3. Scroll up or down.\n4. Wait \n5. Go back\n7. Return to google to start over.\n8. Respond with the final answer\n\nCorrespondingly, Action should STRICTLY follow the format:\n\n- Click [Numerical_Label] \n- Type [Numerical_Label]; [Content] \n- Scroll [Numerical_Label or WINDOW]; [up or down] \n- Wait \n- GoBack\n- Google\n- ANSWER; [content]\n\nKey Guidelines You MUST follow:\n\n* Action guidelines *\n1) Execute only one action per iteration.\n2) When clicking or typing, ensure to select the correct bounding box.\n3) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.\n\n* Web Browsing Guidelines *\n1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages\n2) Select strategically to minimize time wasted.\n\nYour reply should strictly follow the format:\n\nThought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}\nAction: {{One Action format you choose}}\nThen the User will provide:\nObservation: {{A labeled screenshot Given by User}}\n")]), MessagesPlaceholder(variable_name='scratchpad', optional=True), HumanMessagePromptTemplate(prompt=[ImagePromptTemplate(input_variables=['img'], template={'url': 'data:image/png;base64,{img}'}), PromptTemplate(input_variables=['bbox_descriptions'], template='{bbox_descriptions}'), PromptTemplate(input_variables=['input'], template='{input}')])])

llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=4096)
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

# 4. Define graph:
# グラフ状態を更新するhelper関数を定義


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


# create graph

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


for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
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


# 5a. Prepare visualisation for running agent

screenshots = []

import io
import textwrap

from PIL import Image, ImageDraw, ImageFont

base_filename = "agent_path"
path_history_dir = "path-history"
os.makedirs(path_history_dir, exist_ok=True)


def create_text_image(text, width=800, height=100, font_size=100):
    font = ImageFont.load_default()
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    lines = textwrap.wrap(text, width=80)
    y_text = 10
    for line in lines:
        draw.text((10, y_text), line, fill="black", font=font)
        y_text += font_size + 5
    return image


def add_screenshot(image, is_initial=False, is_final=False):
    global screenshots
    if is_initial or is_final:
        screenshots.append(image)
    else:
        screenshots.append(Image.open(image))
    update_agent_path_image(image, is_initial, is_final)


def update_agent_path_image(new_image, is_initial=False, is_final=False):
    global screenshots

    if isinstance(new_image, str):
        img = Image.open(new_image)
    elif isinstance(new_image, Image.Image):
        img = new_image
    else:
        raise ValueError("The new_image parameter must be a file path or a PIL Image object.")

    if is_initial or is_final:
        screenshots.insert(0 if is_initial else len(screenshots), img)
    else:
        screenshots.append(img)
    filename = os.path.join(path_history_dir, f"{base_filename}.png")

    cols = 3
    rows = (len(screenshots) + cols - 1) // cols
    max_width, max_height = get_max_dimensions(screenshots)

    grid_image = Image.new("RGB", (cols * max_width, rows * max_height), color=(255, 255, 255))
    for i, img in enumerate(screenshots):
        x = (i % cols) * max_width
        y = (i // cols) * max_height
        grid_image.paste(img, (x, y))
    grid_image.save(filename)


def get_max_dimensions(screenshots):
    widths, heights = zip(
        *[(img.width, img.height) if isinstance(img, Image.Image) else Image.open(img).size for img in screenshots]
    )
    return max(widths), max(heights)


# 5. Run agent

# エージェントが呼ばれるたびに、ステップを表示し、スクリーンショットを保存する


async def call_agent(question: str, page, max_steps: int = 150):
    objective_image = create_text_image("Objective: " + question, width=800, height=100, font_size=100)
    update_agent_path_image(objective_image, is_initial=True)

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
        update_agent_path_image(img)

        if action and "ANSWER" in action:
            if action_input is None:
                raise ValueError("No answer provided.")
            final_answer = action_input[0]
            # Create and add the final response image
            final_response_image = create_text_image(
                "Final Response: " + final_answer, width=800, height=100, font_size=20
            )
            update_agent_path_image(final_response_image, is_final=True)
            break

    return final_answer


# Main function
async def main(objective: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await stealth_async(page)
        await page.goto("https://www.google.com")

        try:
            res = await call_agent(objective, page)
            print(f"Final response: {res}")
        finally:
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the agent on a given objective")
    parser.add_argument("--objective", type=str, help="The question to run the agent on")
    args = parser.parse_args()
    asyncio.run(main(args.objective))
