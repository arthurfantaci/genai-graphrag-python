"""Knowledge graph builder with schema constraints.

Builds a knowledge graph from PDF documents using a defined schema with node
types, relationship types, and allowed patterns. This approach provides more
consistent entity extraction than the schema-less kg_builder.py.

Overview of Schema-Constrained Knowledge Graph Construction:
------------------------------------------------------------
1. Schema Definition: Define node types (entities), relationship types (edges),
   and valid patterns (allowed source-relationship-target triples).

2. Pipeline Configuration: Configure SimpleKGPipeline with the schema to
   constrain the LLM's entity extraction behavior.

3. Entity Extraction: The LLM receives each text chunk along with the schema
   definitions and attempts to classify entities into the predefined types.

4. Pattern Enforcement: Only relationships matching the PATTERNS list are
   created, preventing semantically invalid connections.

5. Batch Processing: Process all PDF files in a directory, building a unified
   knowledge graph across all documents.

Schema Design Considerations:
-----------------------------
- NODE_TYPES can be simple strings (just labels) or dictionaries with
  descriptions and properties for richer guidance to the LLM.

- RELATIONSHIP_TYPES define the edge labels that connect nodes.

- PATTERNS constrain which node types can be connected by which relationships,
  creating a closed-world assumption that trades recall for precision.

Comparison with Other Builders:
-------------------------------
- kg_builder.py: No schema, LLM has full freedom (less consistent)
- kg_builder_schema.py: Schema-constrained (this file, balanced approach)
- kg_structured_builder.py: Full schema + CSV metadata + Cypher queries (most structured)
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

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
# The LLM receives text chunks along with the schema definitions and extracts
# entities that match the predefined node types.
#
# Key configuration:
#   - model_name: "gpt-4o" provides strong instruction-following for extraction
#   - temperature=0: Deterministic output for consistent entity classification
#   - response_format: Forces JSON output for reliable parsing
# =============================================================================
llm: OpenAILLM = OpenAILLM(
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
# These embeddings enable semantic similarity search in RAG applications.
# =============================================================================
embedder: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# =============================================================================
# TEXT SPLITTER CONFIGURATION
# =============================================================================
# Divides document text into chunks for LLM processing.
#   - chunk_size=500: Characters per chunk (fits in LLM context window)
#   - chunk_overlap=100: Overlapping characters prevent entity loss at boundaries
# =============================================================================
text_splitter: FixedSizeSplitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100)

# =============================================================================
# SCHEMA DEFINITION: NODE TYPES
# =============================================================================
# Node types can be defined in two ways:
#
# 1. Simple strings: Just the label (e.g., "Technology")
#    - LLM uses its general knowledge to classify entities
#    - Quick to define but less precise
#
# 2. Dictionaries with description and properties:
#    - "label": The Neo4j node label
#    - "description": Natural language guidance for the LLM
#    - "properties": Schema for node attributes
#
# This example mixes both approaches to demonstrate flexibility.
# Simple types are good for obvious entities; detailed types help with
# domain-specific or ambiguous categories.
# =============================================================================
NODE_TYPES: list[str | dict[str, Any]] = [
    "Technology",  # Simple string: LLM uses general knowledge
    "Concept",
    "Example",
    "Process",
    "Challenge",
    {
        # Detailed definition with description
        "label": "Benefit",
        "description": "A benefit or advantage of using a technology or approach.",
    },
    {
        # Full definition with properties
        "label": "Resource",
        "description": "A related learning resource such as a book, article, video, or course.",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "type", "type": "STRING"},
        ],
    },
]

# =============================================================================
# SCHEMA DEFINITION: RELATIONSHIP TYPES
# =============================================================================
# Simple string list of allowed relationship types (edge labels).
# The LLM is constrained to only create relationships using these types.
#
# For more guidance, you can use dictionaries with descriptions like NODE_TYPES.
# See kg_structured_builder.py for that approach.
# =============================================================================
RELATIONSHIP_TYPES: list[str] = [
    "RELATED_TO",
    "PART_OF",
    "USED_IN",
    "LEADS_TO",
    "HAS_CHALLENGE",
    "CITES",
]

# =============================================================================
# SCHEMA DEFINITION: ALLOWED PATTERNS
# =============================================================================
# Patterns define valid (source_node_type, relationship_type, target_node_type)
# triples. This is the strictest constraint on the graph structure.
#
# Only relationships matching these patterns will be created. For example:
#   ("Technology", "HAS_CHALLENGE", "Challenge") allows:
#   (:Technology)-[:HAS_CHALLENGE]->(:Challenge)
#
# But (:Challenge)-[:HAS_CHALLENGE]->(:Technology) is NOT allowed.
#
# This enforces semantic correctness: technologies have challenges,
# not the reverse.
# =============================================================================
PATTERNS: list[tuple[str, str, str]] = [
    ("Technology", "RELATED_TO", "Technology"),
    ("Concept", "RELATED_TO", "Technology"),
    ("Example", "USED_IN", "Technology"),
    ("Process", "PART_OF", "Technology"),
    ("Technology", "HAS_CHALLENGE", "Challenge"),
    ("Concept", "HAS_CHALLENGE", "Challenge"),
    ("Technology", "LEADS_TO", "Benefit"),
    ("Process", "LEADS_TO", "Benefit"),
    ("Resource", "CITES", "Technology"),
]

# =============================================================================
# KNOWLEDGE GRAPH BUILDER PIPELINE (WITH SCHEMA)
# =============================================================================
# SimpleKGPipeline configured with a schema dictionary containing:
#   - node_types: Categories of entities to extract
#   - relationship_types: Edge labels for connecting entities
#   - patterns: Valid (source, relationship, target) triples
#
# The schema is passed to the LLM during entity extraction, guiding it to
# classify entities consistently and only create allowed relationships.
# =============================================================================
kg_builder: SimpleKGPipeline = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    embedder=embedder,
    from_pdf=True,
    text_splitter=text_splitter,
    schema={
        "node_types": NODE_TYPES,
        "relationship_types": RELATIONSHIP_TYPES,
        "patterns": PATTERNS,
    },
)

# =============================================================================
# BATCH DOCUMENT PROCESSING
# =============================================================================
# Process all PDF files in the data directory.
#
# Steps for each document:
#   1. Load PDF and extract text
#   2. Split into chunks (500 chars, 100 overlap)
#   3. For each chunk, send to LLM with schema for entity extraction
#   4. Resolve duplicate entities across chunks and documents
#   5. Write nodes and relationships to Neo4j
#   6. Generate and store vector embeddings
#
# Each PDF contributes to a unified knowledge graph, with entities
# potentially connected across documents through entity resolution.
# =============================================================================
data_path: str = "./genai-graphrag-python/data/"
pdf_files: list[str] = [
    os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".pdf")
]

for pdf_file in pdf_files:
    print(f"Processing {pdf_file}")
    result: PipelineResult = asyncio.run(kg_builder.run_async(file_path=pdf_file))
    print(result.result)

# Clean up Neo4j driver connection to release resources
neo4j_driver.close()
