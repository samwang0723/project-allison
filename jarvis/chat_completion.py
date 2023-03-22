import os
import openai
import tiktoken
import pandas as pd

from .constants import ENV_PATH

from dotenv import load_dotenv
from openai.embeddings_utils import get_embedding, cosine_similarity

COMPLETIONS_MODEL = "gpt-3.5-turbo"
ADVANCED_MODEL = "gpt-4"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_SECTION_LEN = 2046
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003
MIN_SIMILARITY = 0.8
SEPARATOR_LEN = len(tiktoken.get_encoding(ENCODING).encode(SEPARATOR))
HEADER = """\n\n---\n\nPlease perform as a professional Crypto.com domain expert that 
can answer questions about Crypto.com specific knowledge giving below context.\n\nContext:\n"""


def _init():
    load_dotenv(dotenv_path=ENV_PATH)
    openai.api_key = os.environ["OPENAI_API_KEY"]


def _params(model: str = COMPLETIONS_MODEL, max_tokens: int = 1024):
    return {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "model": model,
        "top_p": 0.1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }


def chat_completion(
    prompt: str, query: str, model: str = COMPLETIONS_MODEL, max_tokens: int = 1024
):
    return openai.ChatCompletion.create(
        **_params(model=model, max_tokens=max_tokens),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
    )


def construct_prompt(question: str, df: pd.DataFrame):
    most_relevant_document_sections = _order_by_similarity(question, df)

    chosen_sections = []
    chosen_sections_links = []
    chosen_sections_attachments = []
    chosen_sections_len = 0
    similarities = []

    for _, document_section in most_relevant_document_sections.iterrows():
        similarities.append(f"{document_section.title} - {document_section.similarity}")
        if document_section.similarity < MIN_SIMILARITY:
            continue

        chosen_sections_len += int(document_section.num_tokens) + SEPARATOR_LEN
        # if chosen_sections_len > MAX_SECTION_LEN:
        #    break
        if document_section.num_tokens > 4096:
            body = document_section.body[:4096]
        else:
            body = document_section.body

        chosen_sections.append(str(SEPARATOR + body.replace("\n", " ")))
        chosen_sections_links.append(document_section.link)
        chosen_sections_attachments.append(document_section.attachments)

    if len(chosen_sections) == 0:
        prompt = ""
    else:
        prompt = HEADER + "".join(chosen_sections)

    return (prompt, chosen_sections_links, similarities, chosen_sections_attachments)


def inject_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    if "embeddings" in df.columns:
        mask = df["embeddings"].isna()
        df.loc[mask, "embeddings"] = df.loc[mask, "body"].apply(
            lambda x: get_embedding(x, engine=EMBEDDING_MODEL)
        )
    else:
        df["embeddings"] = df["body"].apply(
            lambda x: get_embedding(x, engine=EMBEDDING_MODEL)
        )
    return df


def _order_by_similarity(query: str, df: pd.DataFrame):
    query_embedding = get_embedding(query, engine=EMBEDDING_MODEL)
    df["similarity"] = df.embeddings.apply(
        lambda x: cosine_similarity(x, query_embedding)
    )
    results = df.sort_values("similarity", ascending=False).head(3)
    return results


_init()
