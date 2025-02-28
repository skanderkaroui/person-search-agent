import os
import json
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from app.models.search import SourceInfo
from app.core.config import settings
import urllib.parse
import re
from app.utils.cache import cache

class TwitterSource:
    """Twitter data source using web scraping."""

    @staticmethod
    async def search(query: str, limit: int = 10) -> List[SourceInfo]:
        """
        Search Twitter for information using web scraping.

        Args:
            query: Search query.
            limit: Maximum number of results to return.

        Returns:
            List[SourceInfo]: List of source information.
        """
        # Check cache first
        cached_results = cache.get("Twitter", query)
        if cached_results:
            return [SourceInfo(**item) for item in cached_results]

        try:
            # Encode the query for URL
            encoded_query = urllib.parse.quote(query)

            # Create the search URL
            url = f"https://nitter.net/search?f=tweets&q={encoded_query}"

            # Set headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            }

            # Send the request
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find tweet elements
            tweet_elements = soup.select('.timeline-item')

            sources = []
            for i, tweet in enumerate(tweet_elements[:limit]):
                try:
                    # Extract tweet information
                    username_element = tweet.select_one('.username')
                    fullname_element = tweet.select_one('.fullname')
                    content_element = tweet.select_one('.tweet-content')
                    link_element = tweet.select_one('.tweet-link')

                    if username_element and content_element and link_element:
                        username = username_element.text.strip()
                        fullname = fullname_element.text.strip() if fullname_element else username
                        content = content_element.text.strip()
                        tweet_url = "https://twitter.com" + link_element['href'] if link_element['href'].startswith('/') else link_element['href']

                        # Create source info
                        source_info = SourceInfo(
                            url=tweet_url,
                            title=f"Tweet by {fullname} (@{username})",
                            snippet=content
                        )

                        sources.append(source_info)
                except Exception as e:
                    print(f"Error parsing tweet: {str(e)}")
                    continue

            # Alternative approach if nitter.net doesn't work
            if not sources:
                # Try scraping Twitter directly (limited results without login)
                url = f"https://twitter.com/search?q={encoded_query}&src=typed_query&f=live"
                response = requests.get(url, headers=headers)

                # Extract tweets using regex (basic approach)
                tweet_pattern = r'data-testid="tweet".*?href="(/[^/]+/status/\d+)".*?data-testid="tweetText">(.*?)</div>'
                matches = re.findall(tweet_pattern, response.text, re.DOTALL)

                for i, (tweet_path, content) in enumerate(matches[:limit]):
                    tweet_url = f"https://twitter.com{tweet_path}"
                    clean_content = re.sub(r'<.*?>', '', content).strip()

                    source_info = SourceInfo(
                        url=tweet_url,
                        title=f"Tweet related to {query}",
                        snippet=clean_content
                    )

                    sources.append(source_info)

            # Cache the results
            cache.set("Twitter", query, [source.dict() for source in sources])

            return sources
        except Exception as e:
            print(f"Error scraping Twitter: {str(e)}")
            return []

class GoogleSource:
    """Google data source using web scraping."""

    @staticmethod
    async def search(query: str, limit: int = 5) -> List[SourceInfo]:
        """
        Search Google for information using web scraping.

        Args:
            query: Search query.
            limit: Maximum number of results to return.

        Returns:
            List[SourceInfo]: List of source information.
        """
        # Check cache first
        cached_results = cache.get("Google", query)
        if cached_results:
            return [SourceInfo(**item) for item in cached_results]

        try:
            # Encode the query for URL
            encoded_query = urllib.parse.quote(query)

            # Create the search URL
            url = f"https://www.google.com/search?q={encoded_query}"

            # Set headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            }

            # Send the request
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find search result elements
            search_results = soup.select('.g')

            sources = []
            for i, result in enumerate(search_results[:limit]):
                try:
                    # Extract result information
                    title_element = result.select_one('h3')
                    link_element = result.select_one('a')
                    snippet_element = result.select_one('.VwiC3b')

                    if title_element and link_element:
                        title = title_element.text.strip()
                        link = link_element['href']

                        # Clean the link (remove Google redirects)
                        if link.startswith('/url?'):
                            link = re.search(r'url=([^&]+)', link).group(1)
                            link = urllib.parse.unquote(link)

                        snippet = snippet_element.text.strip() if snippet_element else "No description available"

                        # Create source info
                        source_info = SourceInfo(
                            url=link,
                            title=title,
                            snippet=snippet
                        )

                        sources.append(source_info)
                except Exception as e:
                    print(f"Error parsing Google result: {str(e)}")
                    continue

            # Alternative approach using DuckDuckGo if Google doesn't work well
            if not sources:
                url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
                response = requests.get(url, headers=headers)

                soup = BeautifulSoup(response.text, 'html.parser')
                results = soup.select('.result')

                for i, result in enumerate(results[:limit]):
                    try:
                        title_element = result.select_one('.result__title')
                        link_element = result.select_one('.result__url')
                        snippet_element = result.select_one('.result__snippet')

                        if title_element and link_element:
                            title = title_element.text.strip()
                            link = link_element.text.strip()
                            if not link.startswith('http'):
                                link = 'https://' + link

                            snippet = snippet_element.text.strip() if snippet_element else "No description available"

                            source_info = SourceInfo(
                                url=link,
                                title=title,
                                snippet=snippet
                            )

                            sources.append(source_info)
                    except Exception as e:
                        print(f"Error parsing DuckDuckGo result: {str(e)}")
                        continue

            # Cache the results
            cache.set("Google", query, [source.dict() for source in sources])

            return sources
        except Exception as e:
            print(f"Error scraping Google: {str(e)}")
            return []

# Dictionary mapping source names to their respective classes
SOURCE_MAP = {
    "Twitter": TwitterSource,
    "Google": GoogleSource,
}