from langchain.tools import tool
import requests
from dotenv import load_dotenv
import os
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from rich import print

from bs4 import BeautifulSoup
import trafilatura
import re

try:
    from readability import Document
except ImportError:
    Document = None


load_dotenv()

wrapper = DuckDuckGoSearchAPIWrapper(max_results=10)

search =DuckDuckGoSearchResults(
    num_results=3,
    output_format="list",
    backend="duckduckgo"
)


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information on a topic. Return titles, URLs and snippets of the top results.
    """
    try:
        results = search.run(query)
    except Exception as error:
        return f"Error web_search: {error}"

    out = []

    for result in results:
        out.append(f"[bold blue]{result['title']}[/bold blue]\n{result['link']}\n{result['snippet']}\n")
    if not out:
        return "No good DuckDuckGo Search Result was found"
    return "\n----\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """
    Scrape une page web et retourne son contenu textuel principal.
    """

    if not url or not url.strip():
        return "Error scraping URL: URL vide."

    if not url.startswith(("http://", "https://")):
        return "Error scraping URL: l'URL doit commencer par http:// ou https://."

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15,
            allow_redirects=True,
        )

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" not in content_type:
            return (
                "Error scraping URL: le contenu retourné "
                "n'est pas une page HTML."
            )

        response.encoding = response.apparent_encoding
        html_content = response.text

        # Première méthode : extraction du contenu principal
        extracted_text = trafilatura.extract(
            html_content,
            url=response.url,
            include_comments=False,
            include_tables=True,
            include_links=False,
            favor_precision=True,
        )

        if extracted_text:
            return clean_text(extracted_text)

        # Deuxième méthode : Readability si le module compatible est disponible.
        if Document is not None:
            document = Document(html_content)
            main_html = document.summary()

            soup = BeautifulSoup(main_html, "html.parser")

            for element in soup(
                ["script", "style", "noscript", "svg", "iframe"]
            ):
                element.decompose()

            readability_text = soup.get_text(separator=" ", strip=True)

            if readability_text:
                return clean_text(readability_text)

        # Dernière méthode : récupération de tout le texte visible
        soup = BeautifulSoup(html_content, "html.parser")

        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "iframe",
                "nav",
                "footer",
                "header",
            ]
        ):
            element.decompose()

        page_text = soup.get_text(separator=" ", strip=True)

        if page_text:
            return clean_text(page_text)

        return "Error scraping URL: aucun contenu textuel trouvé."

    except requests.exceptions.Timeout:
        return "Error scraping URL: la requête a dépassé le délai autorisé."

    except requests.exceptions.HTTPError as error:
        status_code = error.response.status_code
        return f"Error scraping URL: erreur HTTP {status_code}."

    except requests.exceptions.ConnectionError:
        return "Error scraping URL: impossible de se connecter au site."

    except requests.exceptions.RequestException as error:
        return f"Error scraping URL: erreur de requête — {error}."

    except Exception as error:
        return f"Error scraping URL: {error}."


def clean_text(text: str) -> str:
    """
    Nettoie les espaces inutiles tout en conservant les paragraphes.
    """

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
