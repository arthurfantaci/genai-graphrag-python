"""Wikipedia article loader example.

Demonstrates how to create a custom DataLoader that fetches content from
Wikipedia articles. This enables building knowledge graphs from Wikipedia's
vast repository of structured knowledge.

Use Cases for Wikipedia Loading:
--------------------------------
- Building domain-specific knowledge graphs from Wikipedia categories
- Enriching existing graphs with Wikipedia context
- Creating educational knowledge bases
- Cross-referencing entities with authoritative sources

How It Works:
-------------
1. Inherit from DataLoader: Standard interface for all loaders
2. Use wikipedia library: Third-party library handles API calls
3. Extract page.content: The plain text of the Wikipedia article
4. Build metadata: Include the Wikipedia URL for attribution

Dependencies:
-------------
Requires the wikipedia package: pip install wikipedia

The wikipedia library handles:
  - Wikipedia API communication
  - Disambiguation (selecting the right article)
  - Content extraction from wiki markup
  - Language selection (defaults to English)
"""

import os

from dotenv import load_dotenv

load_dotenv()

import asyncio
from pathlib import Path
from urllib.parse import quote

import wikipedia
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# tag::import_loader[]
# You will need to install the wikipedia package: pip install wikipedia
from neo4j_graphrag.experimental.components.pdf_loader import (
    DataLoader,
    DocumentInfo,
    PdfDocument,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

# end::import_loader[]

# =============================================================================
# ENVIRONMENT VARIABLE VALIDATION
# =============================================================================
# Validates that all required Neo4j connection parameters are configured.
# Early validation prevents cryptic errors during pipeline execution.
# =============================================================================
NEO4J_URI: str | None = os.getenv("NEO4J_URI")
NEO4J_USERNAME: str | None = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD: str | None = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE: str | None = os.getenv("NEO4J_DATABASE")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE]):
    missing: list[str] = [
        var
        for var, val in [
            ("NEO4J_URI", NEO4J_URI),
            ("NEO4J_USERNAME", NEO4J_USERNAME),
            ("NEO4J_PASSWORD", NEO4J_PASSWORD),
            ("NEO4J_DATABASE", NEO4J_DATABASE),
        ]
        if not val
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Type narrowing: after validation, these are guaranteed to be str
assert NEO4J_URI is not None
assert NEO4J_USERNAME is not None
assert NEO4J_PASSWORD is not None
assert NEO4J_DATABASE is not None

# =============================================================================
# NEO4J CONNECTION
# =============================================================================
# Establish connection to Neo4j database where the knowledge graph will be stored.
# verify_connectivity() ensures the connection is valid before processing begins.
# =============================================================================
neo4j_driver: Driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)
neo4j_driver.verify_connectivity()

# =============================================================================
# LLM AND EMBEDDER CONFIGURATION
# =============================================================================
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "temperature": 0,
        "response_format": {"type": "json_object"},
    },
)

embedder = OpenAIEmbeddings(model="text-embedding-ada-002")


# =============================================================================
# WIKIPEDIA LOADER IMPLEMENTATION
# =============================================================================
# This loader fetches Wikipedia articles by title and returns them as
# PdfDocument objects for the pipeline to process.
#
# Key Features:
#   - filepath parameter is repurposed as the Wikipedia page title
#   - page.content contains the article text (without wiki markup)
#   - URL is constructed for metadata/attribution
#
# Error Handling Considerations:
#   - wikipedia.page() may raise DisambiguationError for ambiguous titles
#   - PageError is raised if the page doesn't exist
#   - Production code should handle these exceptions gracefully
#
# The URL uses quote() to properly encode special characters in the title
# (e.g., spaces become %20).
# =============================================================================
# tag::loader[]
class WikipediaLoader(DataLoader):
    async def run(self, filepath: Path) -> PdfDocument:
        # Load the Wikipedia page
        page = wikipedia.page(filepath)

        # Return a PdfDocument
        return PdfDocument(
            text=page.content,
            document_info=DocumentInfo(
                path=str(filepath),
                metadata={
                    "url": f"https://en.wikipedia.org/w/index.php?title={quote(page.title)}",
                },
            ),
        )


data_loader = WikipediaLoader()
# end::loader[]

# =============================================================================
# PIPELINE CONFIGURATION WITH WIKIPEDIA LOADER
# =============================================================================
# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    embedder=embedder,
    from_pdf=True,
    pdf_loader=data_loader,
)
# end::kg_builder[]

# =============================================================================
# TEST THE LOADER
# =============================================================================
# The file_path parameter is used as the Wikipedia page title.
# This demonstrates the flexibility of the DataLoader interface.
# =============================================================================
# tag::run_loader[]
wikipedia_page = "Knowledge Graph"
doc = asyncio.run(data_loader.run(wikipedia_page))
print(doc.text)
# end::run_loader[]

# =============================================================================
# RUN FULL PIPELINE
# =============================================================================
print(f"Processing {wikipedia_page}")
result = asyncio.run(kg_builder.run_async(file_path=wikipedia_page))
print(result.result)
