"""Standalone entity extraction example.

Demonstrates how to use the LLMEntityRelationExtractor component directly,
outside of the full SimpleKGPipeline. This is useful for:
  - Testing extraction on sample text
  - Debugging extraction quality
  - Integrating extraction into custom pipelines
  - Benchmarking different LLM models or prompts

Component Architecture:
-----------------------
The neo4j-graphrag library uses a component-based architecture where each
step of the KG construction pipeline is a separate, reusable component:

  - DataLoader: Loads documents (PDF, text, web)
  - TextSplitter: Chunks text into smaller pieces
  - LLMEntityRelationExtractor: Extracts entities and relationships
  - EntityResolver: Merges duplicate entities
  - KGWriter: Writes to Neo4j

This example demonstrates using LLMEntityRelationExtractor in isolation,
which is the core component that performs LLM-based extraction.

Input/Output Types:
-------------------
Input: TextChunks containing one or more TextChunk objects
Output: Extracted entities with labels, properties, and relationships

This allows processing pre-chunked text without involving the full pipeline.
"""

from dotenv import load_dotenv

load_dotenv()

import asyncio

from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.llm import OpenAILLM

# =============================================================================
# SAMPLE TEXT FOR EXTRACTION
# =============================================================================
# This text contains several extractable entities and relationships:
#   - Neo4j (Technology)
#   - graph database (Concept)
#   - nodes and relationships (Concept)
#   - labeled property graph (Concept)
#
# The LLM should identify relationships like:
#   - Neo4j IS_A graph database
#   - graph structure CONTAINS nodes and relationships
#   - Neo4j USES labeled property graph
# =============================================================================
text = """
Neo4j is a graph database that stores data in a graph structure.
Data is stored as nodes and relationships instead of tables or documents.
Graph databases are particularly useful when _the connections between data are as important as the data itself_.
A graph shows how objects are related to each other.
The objects are referred to as *nodes* (vertices) connected by *relationships* (edges).
Neo4j uses the graph structure to store data and is known as a *labeled property graph*.
"""

# =============================================================================
# ENTITY RELATION EXTRACTOR CONFIGURATION
# =============================================================================
# The LLMEntityRelationExtractor wraps an LLM for entity extraction.
#
# Configuration:
#   - model_name: The LLM to use for extraction
#   - temperature=0: Deterministic output for consistent extraction
#
# Note: This example uses gpt-4 instead of gpt-4o. Both work, but gpt-4o
# is generally faster and more cost-effective for extraction tasks.
#
# Optional parameters (not shown):
#   - prompt_template: Custom extraction prompt
#   - schema: Node types and relationship types to extract
# =============================================================================
extractor = LLMEntityRelationExtractor(
    llm=OpenAILLM(model_name="gpt-4", model_params={"temperature": 0})
)

# =============================================================================
# RUN EXTRACTION
# =============================================================================
# The extractor expects a TextChunks object containing one or more TextChunk
# objects. Each chunk is processed independently.
#
# TextChunk properties:
#   - text: The text content to extract from
#   - index: Chunk position in the document (used for ordering)
#
# The result contains:
#   - nodes: List of extracted entity nodes with labels and properties
#   - relationships: List of extracted relationships between nodes
#
# This output can be passed to KGWriter for Neo4j population, or
# inspected for debugging and quality assessment.
# =============================================================================
entities = asyncio.run(
    extractor.run(chunks=TextChunks(chunks=[TextChunk(text=text, index=0)]))
)

print(entities)
