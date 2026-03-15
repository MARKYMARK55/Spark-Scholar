"""
title: SearXNG Web Search
author: System
description: Search the web via SearXNG. Nemotron calls this automatically
             when it needs current information. Results are additive context,
             not a replacement for the model's own reasoning.
version: 1.0.0
"""

import httpx
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        searxng_url: str = Field(
            default="http://searxng:8888",
            description="SearXNG base URL"
        )
        max_results: int = Field(
            default=10,
            description="Maximum number of search results to return"
        )
        timeout: int = Field(
            default=10,
            description="Search request timeout in seconds"
        )

    def __init__(self):
        self.valves = self.Valves()

    async def search_web(self, query: str) -> str:
        """
        Search the web for current information using SearXNG.
        Use this when you need up-to-date facts, recent events, news,
        current prices, latest releases, or any information that may
        have changed recently or that you are uncertain about.
        :param query: The search query string
        :return: Search results as formatted text
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.valves.searxng_url}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "language": "en",
                        "categories": "general"
                    },
                    timeout=self.valves.timeout
                )

                if response.status_code != 200:
                    return f"Search failed with status {response.status_code}"

                data = response.json()
                results = data.get("results", [])[:self.valves.max_results]

                if not results:
                    return f"No results found for '{query}'."

                output = f"Web search results for '{query}':\n\n"
                for i, r in enumerate(results, 1):
                    title = r.get("title", "No title")
                    snippet = r.get("content", "No description")
                    url = r.get("url", "")
                    output += f"{i}. {title}\n"
                    output += f"   {snippet}\n"
                    if url:
                        output += f"   Source: {url}\n"
                    output += "\n"

                return output.strip()

        except httpx.TimeoutException:
            return f"Search timed out after {self.valves.timeout}s. Try again or rephrase the query."
        except Exception as e:
            return f"Search failed: {str(e)}"
