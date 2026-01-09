"""Structured knowledge graph builder with schema and CSV-based document processing.

Builds a knowledge graph from multiple PDF documents listed in a CSV file,
using a defined schema with node types, relationship types, and allowed patterns.
Creates additional Lesson nodes linked to Document nodes via Cypher queries.

Overview of the Knowledge Graph Construction Process:
------------------------------------------------------
1. Schema Definition: Define node types (entities), relationship types (edges),
   and valid patterns (source-relationship-target triples) that constrain the
   LLM's entity extraction to produce a consistent, well-structured graph.

2. Pipeline Configuration: Configure SimpleKGPipeline with the schema, LLM,
   embedder, and text splitter. The pipeline orchestrates:
   - PDF text extraction
   - Text chunking for manageable LLM context windows
   - LLM-based entity and relationship extraction guided by the schema
   - Entity resolution to merge duplicate entities
   - Neo4j graph population with extracted nodes and relationships
   - Vector embedding generation for semantic search capabilities

3. Document Processing: Iterate through PDFs listed in a CSV manifest,
   running the pipeline for each document.

4. Supplementary Graph Structure: After extraction, execute Cypher queries
   to create additional domain-specific nodes (Lesson) and relationships
   that link extracted content back to course structure metadata.
"""

from __future__ import annotations

import asyncio
import csv
import os
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase, Record, ResultSummary
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

if TYPE_CHECKING:
    from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult

load_dotenv()

# Environment variable validation
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
# The LLM is the core of entity extraction. It receives text chunks along with
# the schema (node types, relationship types, patterns) and returns structured
# JSON containing identified entities and relationships.
#
# Key configuration:
#   - model_name: "gpt-4o" provides strong instruction-following for extraction
#   - temperature=0: Deterministic output for consistent extraction
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
# The embedder generates vector representations of text chunks. These embeddings
# enable semantic similarity search in the knowledge graph, allowing retrieval
# of relevant chunks based on meaning rather than exact keyword matches.
#
# text-embedding-ada-002 produces 1536-dimensional vectors stored in Neo4j
# for use with VectorCypher retrievers in RAG applications.
# =============================================================================
embedder: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# =============================================================================
# TEXT SPLITTER CONFIGURATION
# =============================================================================
# Divides document text into manageable chunks for LLM processing.
#   - chunk_size=500: Characters per chunk (fits comfortably in LLM context)
#   - chunk_overlap=100: Overlapping characters between consecutive chunks
#
# Overlap is important: entities mentioned at chunk boundaries might be split
# across two chunks. With overlap, the entity appears in full in at least one
# chunk, ensuring it can be properly extracted.
# =============================================================================
text_splitter: FixedSizeSplitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100)

# =============================================================================
# SCHEMA DEFINITION: NODE TYPES
# =============================================================================
# Node types define the categories of entities that the LLM should extract from
# the source documents. Each node type includes:
#   - label: The Neo4j node label (e.g., :Technology, :Concept)
#   - description: Natural language guidance for the LLM to identify this entity type
#   - properties: Schema for node attributes with types and optionality
#
# The LLM uses these descriptions to classify extracted entities. For example,
# when processing text about "Python is a programming language", the LLM
# recognizes "Python" as a Technology node based on the description guidance.
#
# Property definitions enable structured data extraction. The LLM attempts to
# populate each property from the source text context, creating richer nodes
# than simple entity names alone.
# =============================================================================
NODE_TYPES: list[dict[str, Any]] = [
    {
        "label": "Technology",
        "description": "A technology, tool, framework, library, platform, or system.",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "category", "type": "STRING"},
            {"name": "version", "type": "STRING"},
        ],
    },
    {
        "label": "Concept",
        "description": "An abstract idea, principle, theory, methodology, or paradigm.",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "definition", "type": "STRING"},
        ],
    },
    {
        "label": "Example",
        "description": "A specific instance, case study, use case, or demonstration.",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "context", "type": "STRING"},
        ],
    },
    {
        "label": "Process",
        "description": "A workflow, procedure, algorithm, or sequence of steps.",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "steps", "type": "STRING"},
        ],
    },
    {
        "label": "Challenge",
        "description": "A difficulty, limitation, obstacle, risk, or problem to overcome.",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "severity", "type": "STRING"},
        ],
    },
    {
        "label": "Benefit",
        "description": "A benefit, advantage, or positive outcome of using a technology or approach.",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "impact", "type": "STRING"},
        ],
    },
    {
        "label": "Resource",
        "description": "A learning resource such as a book, article, video, course, or documentation.",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "type", "type": "STRING"},
            {"name": "url", "type": "STRING"},
        ],
    },
]

# =============================================================================
# SCHEMA DEFINITION: RELATIONSHIP TYPES
# =============================================================================
# Relationship types define the semantic edge labels that connect nodes in the
# knowledge graph. Each relationship type includes:
#   - label: The Neo4j relationship type (e.g., -[:RELATED_TO]->)
#   - description: Natural language guidance for when to create this relationship
#
# The LLM uses these descriptions to determine how entities relate to each other.
# For example, when text states "Neo4j uses the Cypher query language", the LLM
# creates a RELATED_TO relationship between the Technology nodes.
#
# Well-defined relationship types enable:
#   - Consistent graph traversal patterns
#   - Meaningful path queries (e.g., "What challenges does X have?")
#   - Graph analytics on specific relationship semantics
# =============================================================================
RELATIONSHIP_TYPES: list[dict[str, Any]] = [
    {
        "label": "RELATED_TO",
        "description": "A general association or connection between two entities.",
    },
    {
        "label": "PART_OF",
        "description": "Indicates a component, subset, or hierarchical membership relationship.",
    },
    {
        "label": "USED_IN",
        "description": "Indicates where a technology or concept is applied or utilized.",
    },
    {
        "label": "LEADS_TO",
        "description": "A causal, consequential, or sequential relationship.",
    },
    {
        "label": "HAS_CHALLENGE",
        "description": "Connects an entity to its associated difficulties or limitations.",
    },
    {
        "label": "CITES",
        "description": "References, mentions, or links to another resource or entity.",
    },
]

# =============================================================================
# SCHEMA DEFINITION: ALLOWED PATTERNS
# =============================================================================
# Patterns define the valid (source_node_type, relationship_type, target_node_type)
# triples that the LLM is allowed to create. This is a critical constraint that:
#   1. Prevents semantically invalid relationships (e.g., Challenge -[:PART_OF]-> Benefit)
#   2. Ensures consistent graph structure across all processed documents
#   3. Guides the LLM to focus on domain-relevant connections
#
# Each tuple represents: (SourceLabel, RELATIONSHIP_TYPE, TargetLabel)
#
# Example: ("Technology", "HAS_CHALLENGE", "Challenge") allows:
#   (:Technology {name: "LLMs"})-[:HAS_CHALLENGE]->(:Challenge {name: "Hallucination"})
#
# If a relationship is not listed here, the LLM will not create it, even if
# the source text suggests such a connection. This creates a closed-world
# assumption for the graph schema, trading recall for precision.
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
# KNOWLEDGE GRAPH BUILDER PIPELINE
# =============================================================================
# SimpleKGPipeline is the main orchestrator that coordinates the entire KG
# construction process. Under the hood, it chains together several components:
#
# 1. Document Loader (from_pdf=True):
#    - Extracts raw text from PDF files using PyMuPDF or similar library
#    - Preserves document structure where possible
#
# 2. Text Splitter:
#    - Chunks the extracted text into smaller segments (500 chars, 100 overlap)
#    - Overlap ensures entities at chunk boundaries are captured
#    - Each chunk is processed independently by the LLM
#
# 3. Entity Extractor (LLM-based):
#    - Sends each chunk to the LLM with the schema as context
#    - LLM identifies entities matching NODE_TYPES and extracts properties
#    - LLM identifies relationships matching RELATIONSHIP_TYPES and PATTERNS
#    - Returns structured JSON with nodes and relationships
#
# 4. Entity Resolver:
#    - Merges duplicate entities across chunks (e.g., "GPT-4" and "GPT4")
#    - Uses embedding similarity and string matching heuristics
#    - Prevents graph fragmentation from inconsistent entity naming
#
# 5. Graph Writer:
#    - Converts extracted entities and relationships to Cypher MERGE statements
#    - Creates :Chunk nodes linked to source :Document nodes
#    - Generates vector embeddings for semantic retrieval
#    - Writes everything to Neo4j in a single transaction per chunk
#
# The schema dict constrains extraction to produce a consistent graph structure.
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

data_path: str = "./genai-graphrag-python/data/"

# =============================================================================
# SUPPLEMENTARY GRAPH STRUCTURE: LESSON NODE CREATION
# =============================================================================
# This Cypher query creates additional graph structure that links the automatically
# extracted knowledge graph back to the course curriculum metadata.
#
# Why is this needed?
# -------------------
# The SimpleKGPipeline extracts entities (Technology, Concept, etc.) and creates
# :Document and :Chunk nodes automatically. However, it doesn't know about our
# domain-specific course structure (lessons, modules, courses). This Cypher query
# bridges that gap by:
#
# 1. Finding the :Document node created by the pipeline (matched by file path)
# 2. Creating or merging a :Lesson node with curriculum metadata from the CSV
# 3. Linking Document -> Lesson via [:PDF_OF] relationship
#
# Resulting Graph Structure:
# --------------------------
#   (:Chunk)-[:FROM_DOCUMENT]->(:Document)-[:PDF_OF]->(:Lesson)
#        |                          |
#        v                          v
#   (extracted entities)      (course metadata)
#
# This enables queries like:
#   - "What concepts are taught in Module 3?"
#   - "Which lessons cover Neo4j technology?"
#   - "Show me the learning path for GraphRAG"
#
# MERGE vs CREATE:
# ----------------
# MERGE is used instead of CREATE to handle idempotency. If the same document
# is processed multiple times, the Lesson node won't be duplicated.
# =============================================================================
cypher: str = """
MATCH (d:Document {path: $pdf_path})
MERGE (l:Lesson {url: $url})
SET l.name = $lesson,
    l.module = $module,
    l.course = $course
MERGE (d)-[:PDF_OF]->(l)
"""

# =============================================================================
# DOCUMENT PROCESSING LOOP
# =============================================================================
# Process each document listed in the CSV manifest. The CSV file (docs.csv)
# contains metadata for each PDF:
#   - filename: PDF file name in the data directory
#   - url: Original source URL of the lesson
#   - lesson: Human-readable lesson name
#   - module: Module/section the lesson belongs to
#   - course: Parent course name
#
# For each document, we perform two operations:
#   1. KG Extraction: Run the SimpleKGPipeline to extract entities and relationships
#   2. Metadata Linking: Execute the Cypher query to link Document -> Lesson
# =============================================================================
with open(os.path.join(data_path, "docs.csv"), encoding="utf8", newline="") as f:
    docs_csv: csv.DictReader[str] = csv.DictReader(f)

    for doc in docs_csv:
        # Construct absolute path to PDF file for the pipeline
        doc["pdf_path"] = os.path.join(data_path, doc["filename"])
        print(f"Processing document: {doc['pdf_path']}")

        # ---------------------------------------------------------------------
        # STEP 1: Entity Extraction and Knowledge Graph Population
        # ---------------------------------------------------------------------
        # run_async() executes the full pipeline:
        #   1. Load PDF -> extract text
        #   2. Split text into chunks (500 chars each, 100 char overlap)
        #   3. For each chunk:
        #      a. Send to LLM with schema context for entity extraction
        #      b. Parse LLM response into structured nodes and relationships
        #      c. Resolve entities (merge duplicates across chunks)
        #      d. Write nodes and relationships to Neo4j
        #      e. Generate and store vector embeddings for each chunk
        #
        # The result contains statistics about created nodes and relationships.
        # ---------------------------------------------------------------------
        result: PipelineResult = asyncio.run(
            kg_builder.run_async(file_path=doc["pdf_path"])
        )

        # ---------------------------------------------------------------------
        # STEP 2: Create Supplementary Lesson Node and Link to Document
        # ---------------------------------------------------------------------
        # After the pipeline creates the :Document node (identified by path),
        # we create a :Lesson node with course metadata and link them.
        #
        # The `doc` dictionary is passed as parameters_=doc, which maps CSV
        # columns to Cypher parameters ($pdf_path, $url, $lesson, $module, $course).
        #
        # execute_query() returns:
        #   - records: List of result records (empty for this MERGE operation)
        #   - summary: Execution statistics including counters for created nodes/rels
        #   - keys: Column names in the result (empty here)
        # ---------------------------------------------------------------------
        records: list[Record]
        summary: ResultSummary
        keys: list[str]
        records, summary, keys = neo4j_driver.execute_query(
            cypher, parameters_=doc, database_=NEO4J_DATABASE
        )
        print(result, summary.counters)

# Clean up Neo4j driver connection to release resources
neo4j_driver.close()
