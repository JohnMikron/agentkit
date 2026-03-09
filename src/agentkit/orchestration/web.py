"""
WebAgent for AgentKit.

A high-level agent pre-configured with web search and scraping tools.
"""

from __future__ import annotations

from typing import Any

from agentkit.core.agent import Agent
from agentkit.core.tools import tool

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB_DEPS = True
except ImportError:
    HAS_WEB_DEPS = False


class WebAgent(Agent):
    """
    Agent specialized in web-related tasks.

    Includes built-in tools for:
    - Web search (DuckDuckGo)
    - Web page scraping
    - URL extraction
    """

    def __init__(
        self,
        name: str = "web_assistant",
        model: str | None = None,
        memory: bool = True,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize WebAgent with built-in tools."""
        if not HAS_WEB_DEPS:
            # We don't raise here but we'll warn or let the tools fail if used
            pass

        if system_prompt is None:
            system_prompt = (
                "You are a web research assistant. Your goal is to find accurate "
                "information on the internet. Use your search and scraping tools "
                "to provide well-cited and up-to-date answers."
            )

        super().__init__(
            name=name,
            model=model,
            memory=memory,
            system_prompt=system_prompt,
            **kwargs,
        )

        # Register built-in web tools
        self.add_tool(self.search_web)
        self.add_tool(self.scrape_url)

    @tool
    def search_web(self, query: str, limit: int = 5) -> str:
        """
        Search the web using DuckDuckGo.

        Args:
            query: The search query
            limit: Number of results to return
        """
        if not HAS_WEB_DEPS:
            return "Error: Web dependencies (requests) not installed. pip install requests beautifulsoup4"

        try:
            # Basic DuckDuckGo search implementation using requests
            url = f"https://duckduckgo.com/html/?q={query}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for result in soup.find_all("div", class_="result")[:limit]:
                title = result.find("a", class_="result__a")
                snippet = result.find("a", class_="result__snippet")
                if title and snippet:
                    results.append(f"Title: {title.text.strip()}\nURL: {title['href']}\nSnippet: {snippet.text.strip()}\n")

            if not results:
                return "No results found."

            return "\n".join(results)
        except Exception as e:
            return f"Error searching the web: {e!s}"

    @tool
    def scrape_url(self, url: str) -> str:
        """
        Get the text content of a web page.

        Args:
            url: The URL to scrape
        """
        if not HAS_WEB_DEPS:
            return "Error: Web dependencies (requests) not installed. pip install requests beautifulsoup4"

        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # Simple summarization for very long pages
            if len(text) > 10000:
                text = text[:10000] + "\n... (content truncated)"

            return text
        except Exception as e:
            return f"Error scraping URL: {e!s}"
