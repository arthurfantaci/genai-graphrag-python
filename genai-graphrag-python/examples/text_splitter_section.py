"""Custom section-based text splitter example.

Demonstrates how to create a custom TextSplitter that splits documents
based on section headings rather than character counts. This approach
preserves logical document structure in the knowledge graph.

Why Section-Based Splitting?
----------------------------
Character-based splitting can break text mid-sentence or mid-paragraph,
potentially splitting entities or losing context. Section-based splitting:
  - Preserves logical document structure
  - Keeps related content together
  - Maintains heading hierarchy
  - Produces more coherent chunks for LLM processing

Use Cases:
----------
- Technical documentation with clear section headers
- Books and articles with chapters/sections
- API documentation with endpoint sections
- Legal documents with numbered clauses
- Academic papers with standard sections

How It Works:
-------------
1. Inherit from TextSplitter: Base class defines the interface
2. Implement run(): Process text and return TextChunks
3. Parse headings: Identify section boundaries in the text
4. Create chunks: Each section becomes a separate TextChunk

This example uses AsciiDoc-style headings (== Section Title) but can be
adapted for Markdown (#, ##), HTML (<h1>, <h2>), or any other format.
"""

import os

from dotenv import load_dotenv

load_dotenv()

import asyncio

from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# tag::import_splitter[]
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

# end::import_splitter[]

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
# CUSTOM SECTION-BASED SPLITTER
# =============================================================================
# This splitter divides text at section headings, keeping each section
# as a single chunk. Headings are identified by a configurable prefix.
#
# Constructor:
#   - section_heading: The string that marks section starts (default: "== ")
#
# Algorithm:
#   1. Iterate through lines
#   2. When a heading line is found, save current section as a chunk
#   3. Start accumulating new section content
#   4. After all lines, save the final section
#
# Tradeoffs:
#   - Pros: Preserves document structure, coherent chunks
#   - Cons: Variable chunk sizes (sections may be very short or long)
#
# For very long sections, consider combining with a character-based
# splitter to break large sections into smaller pieces.
# =============================================================================
# tag::splitter[]
class SectionSplitter(TextSplitter):
    def __init__(self, section_heading: str = "== ") -> None:
        self.section_heading = section_heading

    async def run(self, text: str) -> TextChunks:
        index = 0
        chunks = []
        current_section = ""

        for line in text.split("\n"):
            # Does the line start with the section heading?
            if line.startswith(self.section_heading):
                chunks.append(TextChunk(text=current_section, index=index))
                current_section = ""
                index += 1

            current_section += line + "\n"

        # Add the last section
        chunks.append(TextChunk(text=current_section, index=index))

        return TextChunks(chunks=chunks)


splitter = SectionSplitter()
# end::splitter[]

# =============================================================================
# TEST THE SPLITTER
# =============================================================================
# This sample text demonstrates the splitter's behavior with AsciiDoc-style
# headings. Each "==" heading starts a new chunk.
#
# Expected output:
#   - Chunk 0: "= Heading 1\nThis is the main section\n\n"
#   - Chunk 1: "== Sub-heading\nThis is some text.\n\n"
#   - Chunk 2: "== Sub-heading 2\nThis is some more text.\n"
# =============================================================================
# tag::run_splitter[]
text = """
= Heading 1
This is the main section

== Sub-heading
This is some text.

== Sub-heading 2
This is some more text.
"""

chunks = asyncio.run(splitter.run(text))
print(chunks)
# end::run_splitter[]

# =============================================================================
# PIPELINE CONFIGURATION WITH CUSTOM SPLITTER
# =============================================================================
# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    embedder=embedder,
    from_pdf=True,
    text_splitter=splitter,
)
# end::kg_builder[]

# =============================================================================
# RUN PIPELINE
# =============================================================================
pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.pdf"

print(f"Processing {pdf_file}")
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
