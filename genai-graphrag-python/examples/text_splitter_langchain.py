"""LangChain text splitter adapter example.

Demonstrates how to use LangChain's text splitters with the neo4j-graphrag
pipeline. This enables leveraging LangChain's extensive library of splitters
while using neo4j-graphrag for knowledge graph construction.

Why Use LangChain Splitters?
----------------------------
LangChain provides many specialized text splitters:
  - CharacterTextSplitter: Split by character count with separator preference
  - RecursiveCharacterTextSplitter: Hierarchical splitting (paragraphs -> sentences -> words)
  - TokenTextSplitter: Split by token count (important for LLM context limits)
  - MarkdownTextSplitter: Split respecting markdown structure
  - HTMLHeaderTextSplitter: Split HTML by headers
  - PythonCodeTextSplitter: Split Python code respecting function boundaries
  - And many more...

The Adapter Pattern:
--------------------
LangChainTextSplitterAdapter wraps any LangChain text splitter to make it
compatible with the neo4j-graphrag TextSplitter interface:

  LangChain Splitter (input: str -> output: List[str])
                    |
                    v
  LangChainTextSplitterAdapter (input: str -> output: TextChunks)
                    |
                    v
  SimpleKGPipeline text_splitter parameter

This adapter pattern allows mixing LangChain's text processing with
neo4j-graphrag's knowledge graph construction.

Dependencies:
-------------
Requires langchain-text-splitters: pip install langchain-text-splitters
"""

import os

from dotenv import load_dotenv

load_dotenv()

import asyncio

from langchain_text_splitters import CharacterTextSplitter
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# tag::import_splitter[]
# You will need to install langchain-text-splitters: pip install langchain-text-splitters
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

# end::import_splitter[]

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
# LANGCHAIN SPLITTER WITH ADAPTER
# =============================================================================
# This example uses CharacterTextSplitter, but any LangChain splitter works.
#
# CharacterTextSplitter configuration:
#   - separator="\n\n": Prefer splitting on paragraph breaks
#   - chunk_size=500: Target chunk size in characters
#   - chunk_overlap=100: Overlap between consecutive chunks
#
# The adapter wraps the LangChain splitter and converts its output
# (List[str]) to TextChunks, which the pipeline expects.
#
# Alternative LangChain splitters to consider:
#   - RecursiveCharacterTextSplitter: Better for varied text structures
#   - TokenTextSplitter: When you need precise token counts
#   - SpacyTextSplitter: NLP-aware sentence splitting
# =============================================================================
# tag::splitter[]
splitter = LangChainTextSplitterAdapter(
    CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=100,
    )
)
# end::splitter[]

# =============================================================================
# PIPELINE CONFIGURATION WITH LANGCHAIN SPLITTER
# =============================================================================
# The text_splitter parameter accepts any object implementing the
# TextSplitter interface, including our LangChain adapter.
# =============================================================================
# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    embedder=embedder,
    from_pdf=True,
    text_splitter=splitter,
)
# end::kg_builder[]

# =============================================================================
# RUN PIPELINE
# =============================================================================
pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.pdf"

print(f"Processing {pdf_file}")
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
