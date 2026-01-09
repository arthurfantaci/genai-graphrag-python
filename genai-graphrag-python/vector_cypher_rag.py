"""Vector Cypher RAG retriever for semantic search with graph traversal.

Uses vector similarity search on chunk embeddings combined with custom Cypher
queries to retrieve relevant context from the knowledge graph for RAG.

Overview of VectorCypher RAG Pipeline:
--------------------------------------
1. Query Embedding: User's question is converted to a vector embedding using
   the same embedder used during knowledge graph construction.

2. Vector Search: The embedding is used to find the top-k most semantically
   similar text chunks in Neo4j's vector index.

3. Graph Traversal: A custom Cypher query expands from matched chunks to
   retrieve related entities, relationships, and metadata from the graph.

4. Context Assembly: Vector search results plus graph traversal results
   are combined into rich context for answer generation.

5. Answer Generation: An LLM uses the assembled context to generate a
   natural language answer to the original question.

VectorCypher vs Text2Cypher:
----------------------------
- VectorCypher: Best for semantic similarity search where you want to find
  content "similar to" the question. The custom retrieval_query allows
  enriching results with graph context. Works without LLM schema knowledge.

- Text2Cypher: Best for structured queries requiring aggregation, counting,
  or precise traversals. Requires the LLM to generate valid Cypher.

Key Advantage of VectorCypher:
------------------------------
The retrieval_query parameter allows combining vector search with graph
traversal in a single retrieval step. This enables:
  - Enriching chunks with metadata (lesson URL, course info)
  - Following relationships to related entities
  - Aggregating information from connected nodes

This is more powerful than pure vector search because it leverages the
graph structure to provide richer context.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever

if TYPE_CHECKING:
    from neo4j_graphrag.generation.types import RagResultModel

load_dotenv()

# =============================================================================
# ENVIRONMENT VARIABLE VALIDATION
# =============================================================================
# Validates that all required Neo4j connection parameters are configured.
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
# Establish connection to the Neo4j database containing the knowledge graph.
# This database should have:
#   - :Chunk nodes with vector embeddings (created by kg_builder scripts)
#   - A vector index named "chunkVector" on chunk embeddings
#   - Relationships linking chunks to documents and entities
# =============================================================================
neo4j_driver: Driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)
neo4j_driver.verify_connectivity()

# =============================================================================
# EMBEDDER CONFIGURATION
# =============================================================================
# The embedder converts the user's question into a vector for similarity search.
#
# IMPORTANT: Use the same embedding model used during KG construction!
# If chunks were embedded with text-embedding-ada-002, queries must use
# the same model for meaningful similarity scores.
# =============================================================================
embedder: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# =============================================================================
# CUSTOM RETRIEVAL QUERY
# =============================================================================
# This Cypher query executes AFTER vector similarity search and transforms
# the raw chunk matches into rich, contextualized results.
#
# Available variables from vector search:
#   - node: The matched :Chunk node
#   - score: The similarity score (higher = more similar)
#
# This query demonstrates the power of combining vector search with graph
# traversal:
#
# 1. FROM_DOCUMENT traversal: Links chunks back to their source documents
#    and the lessons those documents belong to (for URL attribution).
#
# 2. Entity aggregation: Collects all entities extracted from each chunk
#    and their relationships to other entities, providing graph context
#    beyond the raw text.
#
# The COLLECT subquery gathers related entities into an array, showing:
#   - Entity labels and names
#   - Relationship types
#   - Connected entity information
#
# This rich context helps the LLM generate more informed answers with
# proper attribution to source lessons.
# =============================================================================
retrieval_query = """
MATCH (node)-[:FROM_DOCUMENT]->(d)-[:PDF_OF]->(lesson)
RETURN
    node.text as text, score,
    lesson.url as lesson_url,
    collect {
        MATCH (node)<-[:FROM_CHUNK]-(entity)-[r]->(other)-[:FROM_CHUNK]->()
        WITH toStringList([
            labels(entity)[2],
            entity.name,
            entity.type,
            entity.description,
            type(r),
            labels(other)[2],
            other.name,
            other.type,
            other.description
            ]) as values
        RETURN reduce(acc = "", item in values | acc || coalesce(item || ' ', ''))
    } as associated_entities
"""

# =============================================================================
# VECTORCYPHER RETRIEVER
# =============================================================================
# The VectorCypherRetriever combines vector similarity search with custom
# Cypher traversal:
#
# 1. Embeds the query using the configured embedder
# 2. Searches the "chunkVector" index for similar chunks
# 3. Executes the retrieval_query to enrich results with graph context
# 4. Returns combined results as retriever items
#
# Configuration:
#   - index_name: Name of the Neo4j vector index (created during KG build)
#   - embedder: Model to convert queries to vectors
#   - retrieval_query: Custom Cypher to enrich vector search results
# =============================================================================
retriever: VectorCypherRetriever = VectorCypherRetriever(
    neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    index_name="chunkVector",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

# =============================================================================
# LLM CONFIGURATION FOR ANSWER GENERATION
# =============================================================================
# Unlike Text2Cypher, VectorCypher doesn't use the LLM for query generation.
# The LLM is only used for the final answer generation step.
#
# No temperature=0 is specified here, allowing more creative responses.
# =============================================================================
llm: OpenAILLM = OpenAILLM(model_name="gpt-4o")

# =============================================================================
# GRAPHRAG PIPELINE
# =============================================================================
# GraphRAG combines the VectorCypher retriever with the LLM:
#   1. Retriever finds semantically similar chunks + graph context
#   2. LLM generates answer using the rich context
# =============================================================================
rag: GraphRAG = GraphRAG(retriever=retriever, llm=llm)

# =============================================================================
# QUERY EXECUTION
# =============================================================================
# Execute a sample query demonstrating VectorCypher capabilities.
#
# retriever_config={"top_k": 5} limits vector search to 5 most similar chunks.
# This controls context size and can be tuned based on:
#   - LLM context window limits
#   - Desired answer comprehensiveness
#   - Response latency requirements
#
# return_context=True includes retriever results for debugging/inspection.
# =============================================================================
query_text: str = "Please share all sources available to you where can I learn more about knowledge graphs?"

response: RagResultModel = rag.search(
    query_text=query_text, retriever_config={"top_k": 5}, return_context=True
)

# =============================================================================
# RESPONSE OUTPUT
# =============================================================================
# Display the answer and the context used to generate it.
# The context includes:
#   - Chunk text matched by vector similarity
#   - Lesson URLs for attribution
#   - Associated entities from graph traversal
# =============================================================================
print(response.answer)
if response.retriever_result:
    print("CONTEXT:", response.retriever_result.items)

# Clean up Neo4j driver connection to release resources
neo4j_driver.close()
