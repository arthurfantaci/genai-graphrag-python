import os

from dotenv import load_dotenv

load_dotenv()

import asyncio
from pathlib import Path
from urllib.parse import quote

import wikipedia
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# tag::import_loader[]
# You will need to install the wikipedia package: pip install wikipedia
from neo4j_graphrag.experimental.components.pdf_loader import (
    DataLoader,
    DocumentInfo,
    PdfDocument,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

# end::import_loader[]

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)
neo4j_driver.verify_connectivity()

llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "temperature": 0,
        "response_format": {"type": "json_object"},
    },
)

embedder = OpenAIEmbeddings(model="text-embedding-ada-002")


# tag::loader[]
class WikipediaLoader(DataLoader):
    async def run(self, filepath: Path) -> PdfDocument:
        # Load the Wikipedia page
        page = wikipedia.page(filepath)

        # Return a PdfDocument
        return PdfDocument(
            text=page.content,
            document_info=DocumentInfo(
                path=str(filepath),
                metadata={
                    "url": f"https://en.wikipedia.org/w/index.php?title={quote(page.title)}",
                },
            ),
        )


data_loader = WikipediaLoader()
# end::loader[]

# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    embedder=embedder,
    from_pdf=True,
    pdf_loader=data_loader,
)
# end::kg_builder[]

# tag::run_loader[]
wikipedia_page = "Knowledge Graph"
doc = asyncio.run(data_loader.run(wikipedia_page))
print(doc.text)
# end::run_loader[]

print(f"Processing {wikipedia_page}")
result = asyncio.run(kg_builder.run_async(file_path=wikipedia_page))
print(result.result)
