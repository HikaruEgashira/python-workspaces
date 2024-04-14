import asyncio

from spider_rs import Website  # type: ignore


async def main():
    website = Website("https://choosealicense.com")
    website.crawl()
    print(website.get_links())


asyncio.run(main())
