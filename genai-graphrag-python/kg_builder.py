"""Basic knowledge graph builder using SimpleKGPipeline.

Demonstrates the simplest approach to building a knowledge graph from PDF
documents. This script uses the default pipeline configuration without a
predefined schema, allowing the LLM to freely extract entities and relationships.

Overview of the Knowledge Graph Construction Process:
------------------------------------------------------
1. Document Loading: The pipeline loads and extracts text from PDF files.

2. Text Chunking: Large documents are split into smaller chunks (500 chars)
   with overlap (100 chars) to ensure entities at boundaries are captured.

3. Entity Extraction: Each chunk is sent to an LLM which identifies entities
   (nodes) and relationships (edges) without schema constraints.

4. Entity Resolution: Duplicate entities across chunks are merged to prevent
   graph fragmentation (e.g., "GPT-4" and "GPT4" become one node).

5. Graph Population: Extracted nodes and relationships are written to Neo4j
   using MERGE statements to ensure idempotency.

6. Embedding Generation: Vector embeddings are created for each chunk,
   enabling semantic similarity search for RAG applications.

Note: Without a schema, the LLM has freedom to create any entity types and
relationships, which may result in inconsistent labeling across documents.
For production use, consider using kg_builder_schema.py or kg_structured_builder.py.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

if TYPE_CHECKING:
    from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult

load_dotenv()

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
# LLM CONFIGURATION FOR ENTITY EXTRACTION
# =============================================================================
# The LLM analyzes text chunks and extracts entities and relationships.
# Without a schema, the LLM decides what types of entities to create.
#
# Key configuration:
#   - model_name: "gpt-4o" provides strong extraction capabilities
#   - temperature=0: Deterministic output for reproducible extraction
#   - response_format: Forces JSON output for reliable parsing
# =============================================================================
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "temperature": 0,
        "response_format": {"type": "json_object"},
    },
)

# =============================================================================
# EMBEDDER CONFIGURATION FOR SEMANTIC SEARCH
# =============================================================================
# Generates 1536-dimensional vector embeddings for each text chunk.
# These embeddings are stored in Neo4j and enable semantic similarity search
# for RAG applications using VectorCypherRetriever.
# =============================================================================
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# =============================================================================
# TEXT SPLITTER CONFIGURATION
# =============================================================================
# Divides document text into chunks for LLM processing.
#   - chunk_size=500: Characters per chunk (fits in LLM context window)
#   - chunk_overlap=100: Overlapping characters between consecutive chunks
#
# Overlap ensures entities mentioned at chunk boundaries appear in full in
# at least one chunk, preventing loss of information during extraction.
# =============================================================================
text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100)

# =============================================================================
# KNOWLEDGE GRAPH BUILDER PIPELINE (NO SCHEMA)
# =============================================================================
# SimpleKGPipeline without a schema gives the LLM freedom to extract any
# entity types and relationships it identifies in the text.
#
# This is the simplest configuration but has tradeoffs:
#   Pros:
#   - No upfront schema design required
#   - Can discover unexpected entity types
#   - Good for exploratory analysis
#
#   Cons:
#   - Inconsistent entity labeling across documents
#   - May create redundant relationship types
#   - Harder to query without knowing the schema
#
# For production use cases, prefer schema-constrained pipelines.
# =============================================================================
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    embedder=embedder,
    from_pdf=True,
    text_splitter=text_splitter,
)

# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================
# Process a single PDF file through the pipeline.
#
# run_async() executes the full extraction pipeline:
#   1. Load PDF and extract text
#   2. Split into chunks (500 chars, 100 overlap)
#   3. For each chunk, send to LLM for entity extraction
#   4. Resolve duplicate entities across chunks
#   5. Write nodes and relationships to Neo4j
#   6. Generate and store vector embeddings
#
# The result contains statistics about created nodes and relationships.
# =============================================================================
pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.pdf"
result: PipelineResult = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)

# Clean up Neo4j driver connection to release resources
neo4j_driver.close()
