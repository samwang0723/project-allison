import chromadb
import openai
import uuid
import pandas as pd

from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection

from project_allison.constants import VECTOR_STORAGE


def get_vector_collection(name: str) -> Collection:
    vector_db = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=VECTOR_STORAGE)
    )

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai.api_key, model_name="text-embedding-ada-002"
    )

    return vector_db.get_or_create_collection(name=name, embedding_function=openai_ef)


def insert_vendor_knowledgebase(collection, df: pd.DataFrame):
    for row in df.itertuples():
        collection.add(
            documents=[row.body],
            metadatas=[
                {
                    "title": row.title,
                    "link": row.link,
                    "num_tokens": row.num_tokens,
                    "attachments": " | ".join(row.attachments),
                }
            ],
            ids=[str(uuid.uuid4())],
        )


def query_vector_similarity(collection: Collection, query):
    result = collection.query(query_texts=[query], n_results=3)
    return result
