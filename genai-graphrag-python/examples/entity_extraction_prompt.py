"""Custom entity extraction prompt example.

Demonstrates how to customize the LLM prompt used for entity and relationship
extraction. This allows fine-tuning the extraction behavior for specific
domains or use cases.

Why Customize the Extraction Prompt:
------------------------------------
The default extraction prompt is generic. Customizing it helps:
  - Focus extraction on domain-specific entity types
  - Exclude irrelevant entities (e.g., common words, boilerplate)
  - Add domain context that improves extraction accuracy
  - Enforce specific labeling conventions
  - Guide relationship identification

How It Works:
-------------
1. Import ERExtractionTemplate: The default extraction prompt template
2. Create domain_instructions: Your custom prefix text
3. Combine with DEFAULT_TEMPLATE: Preserves the structured output format
4. Pass to SimpleKGPipeline: Pipeline uses custom prompt for all chunks

Template Structure:
-------------------
The DEFAULT_TEMPLATE includes:
  - Instructions for identifying entities and relationships
  - JSON output format specification
  - Schema integration placeholders
  - Examples of expected output

By prepending domain_instructions, you add context that guides the LLM
before it sees the standard extraction instructions.
"""

import os

from dotenv import load_dotenv

load_dotenv()

import asyncio

from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# tag::import_prompt[]
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm import OpenAILLM

# end::import_prompt[]

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
# CUSTOM EXTRACTION PROMPT
# =============================================================================
# This example adds domain-specific instructions that focus the LLM on
# technology-related entities only.
#
# The domain_instructions are prepended to the DEFAULT_TEMPLATE, which
# contains the core extraction logic and output format specification.
#
# Effective domain instructions should:
#   - Clearly state what entity types to extract
#   - Provide examples of target entities
#   - Specify what to ignore or exclude
#   - Be concise (LLM context is limited)
#
# Alternative Approaches:
#   - Replace DEFAULT_TEMPLATE entirely for full control
#   - Use template.format() to inject variables
#   - Create multiple templates for different document types
# =============================================================================
# tag::prompt[]
domain_instructions = (
    "Only extract entities that are related to the technology industry."
    "These include companies, products, programming languages, frameworks, and tools."
    "\n"
)

prompt_template = ERExtractionTemplate(
    template=domain_instructions + ERExtractionTemplate.DEFAULT_TEMPLATE
)
# end::prompt[]

# =============================================================================
# PIPELINE CONFIGURATION WITH CUSTOM PROMPT
# =============================================================================
# The prompt_template parameter accepts any ERExtractionTemplate instance.
# This prompt will be used for all entity extraction calls during
# document processing.
# =============================================================================
# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    embedder=embedder,
    from_pdf=True,
    prompt_template=prompt_template,
)
# end::kg_builder[]

# =============================================================================
# RUN PIPELINE WITH CUSTOM PROMPT
# =============================================================================
pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.pdf"
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
