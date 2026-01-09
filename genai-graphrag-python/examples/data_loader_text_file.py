"""Plain text file loader example.

Demonstrates how to create a custom DataLoader that reads plain text files
instead of PDFs. This pattern enables building knowledge graphs from any
text-based data source.

Use Cases for Custom Text Loaders:
----------------------------------
- Plain text files (.txt)
- Markdown files (.md)
- Source code files for documentation extraction
- Log files for incident analysis
- CSV/TSV files (with custom parsing)
- Any text-based format not natively supported

How It Works:
-------------
1. Inherit from DataLoader: The base class defines the interface
2. Implement run(): Load and process the text file
3. Return PdfDocument: Despite the name, PdfDocument is the standard
   container for any document's text content

Key Insight - PdfDocument as Generic Container:
-----------------------------------------------
The PdfDocument class is actually a generic text container used throughout
the pipeline, regardless of the original source format. It contains:
  - text: The document's text content
  - document_info: Metadata including path and custom properties

This design allows the same pipeline to process PDFs, text files, web pages,
or any other text source by simply changing the loader.
"""

import os

from dotenv import load_dotenv

load_dotenv()

import asyncio
from pathlib import Path

from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# tag::import_loader[]
from neo4j_graphrag.experimental.components.pdf_loader import (
    DataLoader,
    DocumentInfo,
    PdfDocument,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

# end::import_loader[]

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
# CUSTOM TEXT FILE LOADER
# =============================================================================
# This loader reads plain text files and wraps them in a PdfDocument.
#
# Implementation Details:
#   - Uses utf-8 encoding for broad compatibility
#   - Creates DocumentInfo with the file path as identifier
#   - metadata dict can store custom properties (author, date, etc.)
#
# The run() method is async to match the DataLoader interface, even though
# file reading is synchronous. This allows for future async implementations
# (e.g., reading from cloud storage or APIs).
# =============================================================================
# tag::loader[]
class TextLoader(DataLoader):
    async def run(self, filepath: Path) -> PdfDocument:
        # Process the file
        with open(filepath, encoding="utf-8") as f:
            text = f.read()

        # Return a PdfDocument
        return PdfDocument(
            text=text, document_info=DocumentInfo(path=str(filepath), metadata={})
        )


data_loader = TextLoader()
# end::loader[]

# =============================================================================
# PIPELINE CONFIGURATION WITH TEXT LOADER
# =============================================================================
# Note: from_pdf=True is still required even for text files because it
# triggers the document loading path in the pipeline. The pdf_loader
# parameter specifies which loader class to use.
#
# This naming is a quirk of the current API - "from_pdf" really means
# "load from file using a loader" rather than "load PDF specifically".
# =============================================================================
# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    embedder=embedder,
    from_pdf=True,
    pdf_loader=data_loader,
)
# end::kg_builder[]

# =============================================================================
# TEST THE LOADER INDEPENDENTLY
# =============================================================================
# tag::run_loader[]
pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.txt"
doc = asyncio.run(data_loader.run(pdf_file))
print(doc.text)
# end::run_loader[]

# =============================================================================
# RUN FULL PIPELINE
# =============================================================================
pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.txt"
print(f"Processing {pdf_file}")
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
