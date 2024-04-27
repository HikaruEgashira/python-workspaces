import json

import trafilatura
from attr import dataclass
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)


def fetch_content(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    content = trafilatura.extract(downloaded, include_links=True, output_format="json")
    if content is None:
        return "This page does not contain any text."
    return content


def summerize(text: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                task: summerize
                locale: ja
                format: markdown table
                columns:
                    who: VARCHAR(255)
                    when: VARCHAR(255)
                    what: VARCHAR(255)
                    where: VARCHAR(255)
                    why: VARCHAR(255)
                    how: VARCHAR(255)
                    academic category: VARCHAR(255)
                    url: VARCHAR(255)
          """,
            ),
            ("user", "{input}"),
        ]
    )
    chain = prompt | llm
    message = chain.invoke({"input": text})
    if not isinstance(message.content, str):
        raise ValueError("Invalid response from model")
    return message.content


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

    text = fetch_content(arg.url)
    print(summerize(text))
