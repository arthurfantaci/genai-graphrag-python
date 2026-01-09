"""Custom PDF loader example with text preprocessing.

Demonstrates how to extend the built-in PdfLoader to add custom text
preprocessing logic. This is useful when PDFs contain artifacts, formatting
characters, or markup that should be removed before entity extraction.

Use Cases for Custom PDF Loaders:
---------------------------------
- Removing headers/footers from each page
- Stripping markup language syntax (AsciiDoc, Markdown, etc.)
- Normalizing whitespace and line breaks
- Extracting only specific sections of PDFs
- Adding custom metadata based on PDF content
- Handling encrypted or password-protected PDFs

How It Works:
-------------
1. Inherit from PdfLoader: The base class handles the PDF parsing
2. Override run(): Call super().run() to get the parsed document
3. Process pdf_document.text: Apply custom transformations
4. Return modified document: Pipeline uses the cleaned text

In this example, we remove AsciiDoc attribute lines (like :id: value)
that may appear in PDFs generated from AsciiDoc source files.
"""

import os

from dotenv import load_dotenv

load_dotenv()

import asyncio
import re
from pathlib import Path

from fsspec import AbstractFileSystem
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# tag::import_loader[]
from neo4j_graphrag.experimental.components.pdf_loader import PdfDocument, PdfLoader
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
# CUSTOM PDF LOADER IMPLEMENTATION
# =============================================================================
# This class extends PdfLoader to add custom text preprocessing.
#
# Key Pattern:
#   1. Call super().run() to get the default PDF parsing behavior
#   2. Modify pdf_document.text with custom transformations
#   3. Return the modified document
#
# The regex pattern `:*:.*\n?` matches AsciiDoc attribute definitions:
#   - :id: some-value
#   - :author: John Doe
#   - ::term:: definition (also matched)
#
# These attributes are metadata in the source AsciiDoc files but appear
# as noise in the extracted PDF text.
# =============================================================================
# tag::loader[]
class CustomPDFLoader(PdfLoader):
    async def run(
        self,
        filepath: str | Path,
        metadata: dict[str, str] | None = None,
        fs: AbstractFileSystem | str | None = None,
    ) -> PdfDocument:
        pdf_document = await super().run(filepath, metadata, fs)

        # Process the PDF document
        # remove asciidoc attribute lines like :id:
        pdf_document.text = re.sub(
            r":*:.*\n?", "", pdf_document.text, flags=re.MULTILINE
        )

        return pdf_document


data_loader = CustomPDFLoader()
# end::loader[]

# =============================================================================
# PIPELINE CONFIGURATION WITH CUSTOM LOADER
# =============================================================================
# The pdf_loader parameter accepts any class that implements the DataLoader
# interface. By passing our CustomPDFLoader, the pipeline will use our
# preprocessing logic for all PDF files.
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

# tag::pdf_file[]
pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.pdf"
# end::pdf_file[]

# =============================================================================
# TEST THE LOADER INDEPENDENTLY
# =============================================================================
# Running the loader directly allows inspection of the preprocessed text
# before it enters the full pipeline. This is useful for debugging and
# validating that the preprocessing works correctly.
# =============================================================================
# tag::run_loader[]
doc = asyncio.run(data_loader.run(pdf_file))
print(doc.text)
# end::run_loader[]

# =============================================================================
# RUN FULL PIPELINE
# =============================================================================
print(f"Processing {pdf_file}")
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
