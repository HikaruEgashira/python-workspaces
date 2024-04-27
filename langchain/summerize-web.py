import trafilatura
from attr import dataclass
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0)


def fetch_content(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    content = trafilatura.extract(downloaded)
    if content is None:
        return "This page does not contain any text."
    return content


def summerize(text: str) -> str | list[str | dict[str, str]]:
    prompt = ChatPromptTemplate.from_messages([("system", "summerize using 5w1h, outline format"), ("user", text)])
    chain = prompt | llm
    message = chain.invoke({"input": text})
    # if type(message) != str:
    #     raise ValueError("Invalid response from model")
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
