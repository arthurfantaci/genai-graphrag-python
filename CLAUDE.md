# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository accompanies the [Constructing Knowledge Graphs with Neo4j GraphRAG for Python course](https://graphacademy.neo4j.com/courses/genai-graphrag-python/) on GraphAcademy. It demonstrates how to build knowledge graphs from unstructured data using the `neo4j-graphrag` Python library.

## Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run environment verification
uv run python genai-graphrag-python/test_environment.py

# Run any Python script
uv run python <script_path>

# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Lint and auto-fix
uv run ruff check --fix .
```

## Environment Setup

Copy `.env.example` to `.env` and configure:
- `OPENAI_API_KEY` - OpenAI API key
- `NEO4J_URI` - Neo4j connection URI
- `NEO4J_USERNAME` / `NEO4J_PASSWORD` - Neo4j credentials
- `NEO4J_DATABASE` - Target database name

## Architecture

### Core Pipeline Scripts (genai-graphrag-python/)

| Script | Purpose |
|--------|---------|
| `kg_builder.py` | Basic knowledge graph construction using `SimpleKGPipeline` |
| `kg_builder_schema.py` | KG construction with schema constraints (node types, text splitter) |
| `kg_structured_builder.py` | Full schema with node types, relationship types, and patterns; processes multiple PDFs from `docs.csv` |
| `extract_schema.py` | Schema extraction from text using LLM |
| `text2cypher_rag.py` | RAG using Text2Cypher retriever (natural language to Cypher) |
| `vector_cypher_rag.py` | RAG using VectorCypher retriever (semantic search) |

### Examples (genai-graphrag-python/examples/)

Demonstrates customization points:
- `data_loader_*.py` - Custom PDF loaders, text file loading, Wikipedia integration
- `text_splitter_*.py` - LangChain text splitter adapter, section-based splitting
- `entity_extraction_prompt.py` - Custom extraction prompts
- `lexical_graph_config.py` - Lexical graph configuration
- `no_entity_resolution.py` - Disabling entity resolution

### Solutions (genai-graphrag-python/solutions/)

Complete implementations of course exercises.

## Key Patterns

All scripts follow this structure:
1. Load environment with `dotenv`
2. Create Neo4j driver and verify connectivity
3. Configure OpenAI LLM and embedder
4. Build pipeline (`SimpleKGPipeline` for KG construction, `GraphRAG` for retrieval)
5. Run async with `asyncio.run()`

Primary dependencies: `neo4j-graphrag`, `neo4j`, `openai`, `python-dotenv`

## Data

PDF documents in `genai-graphrag-python/data/` cover GenAI fundamentals and are indexed via `docs.csv` for batch processing.
