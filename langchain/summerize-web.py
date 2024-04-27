import trafilatura
from attr import dataclass
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

set_llm_cache(SQLiteCache())


def fetch_content(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    content = trafilatura.extract(downloaded, include_links=True, output_format="json")
    if content is None:
        return "This page does not contain any content."

    return content


class SummarizeOutput(BaseModel):
    Who: str = Field(description="who")
    When: str = Field(description="when")
    What: str = Field(description="what")
    Where: str = Field(description="where")
    Why: str = Field(description="why")
    How: str = Field(description="how")
    Category: str = Field(description="academic category")
    URL: str = Field(description="url")


def summarize(content: str) -> SummarizeOutput:
    llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    parser = PydanticOutputParser(pydantic_object=SummarizeOutput)
    prompt = PromptTemplate(
        template="""
        task: Summarize in japanese
        format:
            {format_instructions}
        content:
            {content}
        """,
        input_variables=["content"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    message = chain.invoke({"content": content})
    if not isinstance(message, SummarizeOutput):
        raise ValueError("The output is not of the expected type.")

    return message


@dataclass
class Args:
    url: str | None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Webページを要約します")
    parser.add_argument("url", type=str, help="要約するWebページのURL", nargs="?")
    arg = Args(**vars(parser.parse_args()))
    if arg.url is None:
        parser.print_help()
        exit(0)

    content = fetch_content(arg.url)
    summary = summarize(content)
    print(summary.model_dump_json())
