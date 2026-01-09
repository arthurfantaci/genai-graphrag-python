"""Schema extraction from unstructured text using LLM.

Demonstrates how to automatically infer a knowledge graph schema from raw text
using the SchemaFromTextExtractor component. This is useful when you don't have
a predefined schema and want the LLM to suggest entity types and relationships
based on the content.

Overview of Schema Extraction Process:
--------------------------------------
1. Text Input: Provide raw, unstructured text containing information about
   entities and their relationships.

2. LLM Analysis: The SchemaFromTextExtractor sends the text to an LLM with
   instructions to identify:
   - Entity types (potential node labels like Person, Organization, Technology)
   - Relationship types (potential edge labels like WORKS_FOR, USES, RELATED_TO)
   - Suggested patterns (valid source-relationship-target triples)

3. Schema Output: Returns a structured schema that can be used with
   SimpleKGPipeline for consistent entity extraction across documents.

Use Cases:
----------
- Exploratory analysis of new document collections
- Bootstrapping schema design before manual refinement
- Validating intuitions about what entities exist in a corpus
"""

from dotenv import load_dotenv

load_dotenv()

import asyncio

from neo4j_graphrag.experimental.components.schema import SchemaFromTextExtractor
from neo4j_graphrag.llm import OpenAILLM

# =============================================================================
# SCHEMA EXTRACTOR CONFIGURATION
# =============================================================================
# SchemaFromTextExtractor uses an LLM to analyze text and propose a schema.
# The LLM identifies:
#   - Potential node types based on entities mentioned in the text
#   - Potential relationship types based on how entities interact
#   - Patterns showing which relationships connect which node types
#
# Configuration:
#   - model_name: "gpt-4o" provides strong reasoning for schema inference
#   - temperature=0: Deterministic output for consistent schema extraction
# =============================================================================
schema_extractor = SchemaFromTextExtractor(
    llm=OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
)

# =============================================================================
# SAMPLE TEXT FOR SCHEMA EXTRACTION
# =============================================================================
# This text serves as input for the schema extractor. The LLM will analyze it
# to identify:
#   - "Neo4j" as a potential Technology or Product node
#   - "graph database management system" as a potential Concept or Category
#   - "Neo4j Inc" as a potential Organization or Company node
#   - Relationships like DEVELOPED_BY, IS_A, or CREATED_BY
#
# In practice, you would use representative samples from your document corpus
# to help the LLM understand what entities and relationships exist.
# =============================================================================
text = """
Neo4j is a graph database management system (GDBMS) developed by Neo4j Inc.
"""

# =============================================================================
# SCHEMA EXTRACTION EXECUTION
# =============================================================================
# run() analyzes the input text and returns a schema dictionary containing:
#   - node_types: List of identified entity categories with descriptions
#   - relationship_types: List of identified relationship types with descriptions
#   - patterns: List of (source, relationship, target) triples
#
# This extracted schema can then be passed to SimpleKGPipeline to constrain
# entity extraction when processing the full document corpus.
# =============================================================================
extracted_schema = asyncio.run(schema_extractor.run(text=text))

print(extracted_schema)
