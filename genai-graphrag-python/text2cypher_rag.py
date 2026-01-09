"""Text2Cypher RAG retriever for natural language to Cypher query conversion.

Uses an LLM to convert natural language questions into Cypher queries,
executes them against the knowledge graph, and returns results for RAG.

Overview of Text2Cypher RAG Pipeline:
-------------------------------------
1. Query Reception: User provides a natural language question about the
   knowledge graph (e.g., "How many technologies are mentioned?").

2. Cypher Generation: The Text2CypherRetriever sends the question to an LLM
   along with optional examples. The LLM generates a Cypher query that would
   answer the question.

3. Query Execution: The generated Cypher query is executed against Neo4j,
   retrieving relevant data from the knowledge graph.

4. Context Assembly: Query results are formatted as context for the final
   response generation.

5. Answer Generation: A second LLM call uses the retrieved context to
   generate a natural language answer to the original question.

Text2Cypher vs VectorCypher:
----------------------------
- Text2Cypher: Best for structured queries where the answer requires
  aggregation, counting, or precise graph traversal. Requires the LLM to
  understand the graph schema to generate valid Cypher.

- VectorCypher: Best for semantic similarity search where you want to find
  chunks "similar to" the question. Works even without schema knowledge.

Example Use Cases for Text2Cypher:
----------------------------------
- "How many nodes of type Technology exist?"
- "What challenges are associated with LLMs?"
- "List all resources that cite Neo4j"
- "Find the path between concept A and technology B"
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever

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
# This is the same database populated by the kg_builder scripts.
# =============================================================================
neo4j_driver: Driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)
neo4j_driver.verify_connectivity()

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
# The LLM serves two purposes in Text2Cypher RAG:
#
# 1. Cypher Generation: Converts natural language questions into Cypher queries.
#    This requires understanding the graph schema and Cypher syntax.
#
# 2. Answer Generation: Uses retrieved context to generate the final answer.
#
# Key configuration:
#   - temperature=0: Deterministic output for consistent Cypher generation
#   - No response_format: The LLM generates plain text Cypher, not JSON
# =============================================================================
llm: OpenAILLM = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

# =============================================================================
# FEW-SHOT EXAMPLES FOR CYPHER GENERATION
# =============================================================================
# Examples help the LLM understand:
#   - The graph schema (what node labels and properties exist)
#   - How to translate natural language patterns to Cypher patterns
#   - Parameter usage with $variable syntax
#
# Each example is a string showing:
#   - USER INPUT: The natural language question
#   - QUERY: The corresponding Cypher query
#
# More examples lead to better Cypher generation, especially for:
#   - Domain-specific query patterns
#   - Complex traversals
#   - Aggregations and filtering
#
# The LLM uses these examples via in-context learning to generate queries
# for new questions that follow similar patterns.
# =============================================================================
examples: list[str] = [
    "USER INPUT: 'Find a node with the name $name?' "
    "QUERY: MATCH (node) WHERE toLower(node.name) CONTAINS toLower($name) "
    "RETURN node.name AS name, labels(node) AS labels",
]

# =============================================================================
# TEXT2CYPHER RETRIEVER
# =============================================================================
# The Text2CypherRetriever orchestrates the query generation and execution:
#
# 1. Takes a natural language question
# 2. Sends it to the LLM with the examples as few-shot context
# 3. LLM generates a Cypher query
# 4. Executes the query against Neo4j
# 5. Returns results as retriever items for the RAG pipeline
#
# The retriever automatically injects graph schema information to help the
# LLM generate valid queries for the specific database structure.
# =============================================================================
retriever: Text2CypherRetriever = Text2CypherRetriever(
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    llm=llm,
    examples=examples,
)

# =============================================================================
# GRAPHRAG PIPELINE
# =============================================================================
# GraphRAG combines the retriever with an LLM for end-to-end question answering:
#
# 1. Retriever: Fetches relevant context from the knowledge graph
# 2. LLM: Generates a natural language answer using the context
#
# This separation allows using different retrievers (Text2Cypher, VectorCypher)
# while keeping the same answer generation logic.
# =============================================================================
rag: GraphRAG = GraphRAG(retriever=retriever, llm=llm)

# =============================================================================
# QUERY EXECUTION
# =============================================================================
# Execute a sample query demonstrating Text2Cypher capabilities.
#
# The search() method:
#   1. Passes query_text to the retriever
#   2. Retriever generates and executes Cypher
#   3. Retrieved results become context for answer generation
#   4. LLM generates final answer using the context
#
# return_context=True includes the retriever results in the response,
# allowing inspection of:
#   - The generated Cypher query (in metadata)
#   - The raw query results (in items)
# =============================================================================
query_text: str = "How many technologies are mentioned in the knowledge graph?"

response: RagResultModel = rag.search(query_text=query_text, return_context=True)

# =============================================================================
# RESPONSE OUTPUT
# =============================================================================
# Display the answer and debugging information:
#   - answer: The LLM-generated natural language response
#   - cypher: The Cypher query generated by Text2CypherRetriever
#   - items: Raw query results from Neo4j
# =============================================================================
print(response.answer)
if response.retriever_result:
    if response.retriever_result.metadata:
        print("CYPHER :", response.retriever_result.metadata.get("cypher"))
    print("CONTEXT:", response.retriever_result.items)

# Clean up Neo4j driver connection to release resources
neo4j_driver.close()
