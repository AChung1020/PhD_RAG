import asyncio

import aiohttp

from PhD_RAG.src.config import settings


async def cli_query():
    async with aiohttp.ClientSession() as session:
        while True:
            query = input("Enter query (type 'exit' to quit): ")
            if query.lower() == "exit":
                print("Exiting chatbot")
                break

            payload = {"query": query}
            try:
                async with session.post(settings.api_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"Claude: {data.get('answer')}")
                    else:
                        print(f"Error: {response.status}")
            except aiohttp.ClientError as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(cli_query())
