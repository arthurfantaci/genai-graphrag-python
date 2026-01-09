"""Lexical graph configuration example.

Demonstrates how to customize the structure of the lexical graph created
by SimpleKGPipeline. The lexical graph is the underlying document structure
that links chunks to documents and tracks chunk ordering.

What is the Lexical Graph?
--------------------------
The lexical graph provides the structural backbone of the knowledge graph:

    (:Document)-[:HAS_CHUNK]->(:Chunk)-[:NEXT]->(:Chunk)
                                  |
                                  v
                            (:Entity)-[:RELATED_TO]->(:Entity)

Default node labels and relationship types:
  - Document nodes: Store document metadata
  - Chunk nodes: Store text chunks with embeddings
  - NEXT relationships: Link sequential chunks
  - FROM_CHUNK relationships: Link entities to their source chunks

Why Customize the Lexical Graph:
--------------------------------
Customization enables:
  - Domain-specific naming (e.g., "Lesson" instead of "Document")
  - Integration with existing graph schemas
  - Custom embedding property names for multiple vector indexes
  - Meaningful relationship types for your domain

Configuration Options:
----------------------
LexicalGraphConfig provides these customization points:
  - chunk_node_label: Label for chunk nodes (default: "Chunk")
  - document_node_label: Label for document nodes (default: "Document")
  - chunk_to_document_relationship_type: Chunk->Document edge (default: "FROM_DOCUMENT")
  - next_chunk_relationship_type: Sequential chunk linking (default: "NEXT_CHUNK")
  - node_to_chunk_relationship_type: Entity->Chunk edge (default: "FROM_CHUNK")
  - chunk_embedding_property: Property name for embeddings (default: "embedding")
"""

import os

from dotenv import load_dotenv

load_dotenv()

import asyncio

from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# tag::import_config[]
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

# end::import_config[]

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
# LEXICAL GRAPH CONFIGURATION
# =============================================================================
# This configuration customizes the lexical graph to use domain-specific
# terminology for an educational content knowledge graph.
#
# Resulting Graph Structure:
#   (:Lesson)-[:IN_LESSON]->(:Section)-[:NEXT_SECTION]->(:Section)
#                                |
#                                v
#                          (:Entity)-[:IN_SECTION]->(:Section)
#
# Property customization:
#   - embeddings: Custom property name for vector embeddings
#     (useful when supporting multiple embedding models)
#
# Note: The vector index must be created on the custom property name
# if using a non-default embedding property.
# =============================================================================
# tag::config[]
config = LexicalGraphConfig(
    chunk_node_label="Section",
    document_node_label="Lesson",
    chunk_to_document_relationship_type="IN_LESSON",
    next_chunk_relationship_type="NEXT_SECTION",
    node_to_chunk_relationship_type="IN_SECTION",
    chunk_embedding_property="embeddings",
)
# end::config[]

# =============================================================================
# PIPELINE CONFIGURATION WITH CUSTOM LEXICAL GRAPH
# =============================================================================
# The lexical_graph_config parameter applies to all graph writes.
# The pipeline will use "Section" and "Lesson" labels instead of
# "Chunk" and "Document".
# =============================================================================
# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    embedder=embedder,
    from_pdf=True,
    lexical_graph_config=config,
)
# end::kg_builder[]

# =============================================================================
# RUN PIPELINE
# =============================================================================
pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.pdf"
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
