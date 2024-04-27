import argparse
import warnings

import trafilatura
from langchain.cache import SQLiteCache
from langchain_anthropic import ChatAnthropic
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


def fetch_content(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    content = trafilatura.extract(downloaded, include_links=True, output_format="json")
    if content is None:
        raise ValueError("Failed to extract content from the URL.")
    return content


class SummarizeOutput(BaseModel):
    Who: str = Field(description="who")
    When: str = Field(description="when")
    What: str = Field(description="what")
    Where: str = Field(description="where")
    Why: str = Field(description="why")
    How: str = Field(description="how")
    URL: str = Field(description="url")


def summarize(content: str) -> SummarizeOutput:
    llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
    structured_llm = llm.with_structured_output(SummarizeOutput)
    prompt = PromptTemplate(
        template="""
        ----task
        Summarize in japanese
        ----content
        {content}
        """,
        input_variables=["content"],
    )
    chain = prompt | structured_llm
    result = chain.invoke({"content": content})
    if not isinstance(result, SummarizeOutput):
        raise ValueError("The output is not a valid SummarizeOutput object.")
    return result


warnings.filterwarnings("ignore")
set_llm_cache(SQLiteCache())

parser = argparse.ArgumentParser(description="Webページを要約します")
parser.add_argument("url", type=str, help="要約するWebページのURL", nargs="?")
arg = parser.parse_args()
if arg.url is None:
    parser.print_help()
    exit(0)

content = fetch_content(arg.url)
summary = summarize(content)
print(summary.json(indent=4, ensure_ascii=False))
