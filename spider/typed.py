import asyncio
from dataclasses import dataclass

from spider_rs import crawl  # type: ignore


@dataclass
class Page:
    """
    a simple page object
    https://github.com/spider-rs/spider-py/blob/main/src/npage.rs#L11
    """

    url: str
    content: str
    status_code: int
    raw_content: bytes


@dataclass
class Website:
    links: list[str]
    pages: list[Page]


def to_website(website) -> Website:
    return Website(
        website.links,
        [Page(page.url, page.content, page.status_code, page.raw_content) for page in website.pages],
    )


async def main():
    website = to_website(await crawl("https://jeffmendez.com"))
    print(website.links)
    print(website.pages)


asyncio.run(main())
