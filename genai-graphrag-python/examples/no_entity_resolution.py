"""Disabling entity resolution example.

Demonstrates how to disable entity resolution in the knowledge graph
construction pipeline. Entity resolution is the process of merging
duplicate entities that refer to the same real-world concept.

What is Entity Resolution?
--------------------------
When processing multiple text chunks, the same entity may be extracted
multiple times with slight variations:
  - "GPT-4" vs "GPT4" vs "GPT 4"
  - "Neo4j" vs "Neo4J" vs "neo4j"
  - "Knowledge Graph" vs "knowledge graphs"

Entity resolution identifies these as the same entity and merges them
into a single node in the graph, preventing fragmentation.

Why Disable Entity Resolution?
------------------------------
You might want to disable entity resolution when:

1. Speed is critical: Entity resolution adds processing time, especially
   for large documents with many entities.

2. Preserving variations: You want to keep entity variations separate
   for analysis (e.g., studying how terminology evolves).

3. High-confidence extraction: Your schema is precise enough that
   duplicates are unlikely.

4. Post-processing: You plan to do custom entity resolution later
   using graph algorithms or external services.

5. Debugging: Inspecting raw extraction output before resolution.

Tradeoffs:
----------
- With resolution: Cleaner graph, fewer nodes, but some false merges possible
- Without resolution: More nodes, but no accidental merges of distinct entities
"""

import os

from dotenv import load_dotenv

load_dotenv()

import asyncio

from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

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
# PIPELINE WITHOUT ENTITY RESOLUTION
# =============================================================================
# Setting perform_entity_resolution=False disables the entity resolution
# component in the pipeline.
#
# Pipeline flow WITH resolution (default):
#   Chunks -> Extract -> Resolve -> Write to Neo4j
#
# Pipeline flow WITHOUT resolution:
#   Chunks -> Extract -> Write to Neo4j
#
# Impact on the graph:
#   - Duplicate entities will create separate nodes
#   - Each chunk's entities are written as-is
#   - Relationships may connect to different versions of the same entity
#
# This can result in a larger, more fragmented graph, but guarantees
# that extracted entities are preserved exactly as the LLM returned them.
# =============================================================================
# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    embedder=embedder,
    from_pdf=True,
    perform_entity_resolution=False,
)
# end::kg_builder[]

# =============================================================================
# RUN PIPELINE
# =============================================================================
pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.pdf"
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
