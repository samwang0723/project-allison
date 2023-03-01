#!/usr/bin/env python3

import os
import openai
import numpy as np
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from termcolor import colored


class KnowledgeBase:
    COMPLETIONS_MODEL = "text-davinci-003"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": 0.0,
        "max_tokens": 300,
        "model": COMPLETIONS_MODEL,
    }
    MAX_SECTION_LEN = 500
    SEPARATOR = "\n* "
    ENCODING = "gpt2"  # encoding for text-davinci-003

    def __init__(self, api_key):
        openai.api_key = api_key
        self.prompt = ""

    def load_dataset(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.set_index(["title", "heading"])
        return df

    def get_embedding(self, text: str, model: str = EMBEDDING_MODEL) -> list[float]:
        result = openai.Embedding.create(model=model, input=text)
        return result["data"][0]["embedding"]

    def compute_doc_embeddings(
        self, df: pd.DataFrame
    ) -> dict[tuple[str, str], list[float]]:
        return {idx: self.get_embedding(r.content) for idx, r in df.iterrows()}

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def vector_similarity(self, x: list[float], y: list[float]) -> float:
        return np.dot(np.array(x), np.array(y))

    def order_document_sections_by_query_similarity(
        self, query: str, contexts: dict[tuple[str, str], np.array]
    ) -> list[tuple[float, tuple[str, str]]]:
        query_embedding = self.get_embedding(query)

        document_similarities = sorted(
            [
                (self.vector_similarity(query_embedding, doc_embedding), doc_index)
                for doc_index, doc_embedding in contexts.items()
            ],
            reverse=True,
        )

        return document_similarities

    def construct_prompt(
        self, question: str, context_embeddings: dict, df: pd.DataFrame
    ) -> str:
        most_relevant_document_sections = (
            self.order_document_sections_by_query_similarity(
                question, context_embeddings
            )
        )

        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []

        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.
            df2 = df.sort_index()
            document_section = df2.loc[section_index]

            # chosen_sections_len += document_section.tokens + separator_len
            # if chosen_sections_len > MAX_SECTION_LEN:
            #    break

            chosen_sections.append(
                str(self.SEPARATOR + document_section.content.replace("\n", " "))
            )
            chosen_sections_indexes.append(str(section_index))

        # Useful diagnostic information
        # print(f"Selected {len(chosen_sections)} document sections:")
        # print("\n".join(chosen_sections_indexes))

        header = """Answer the question as truthfully as possible using the provided 
        context, and if the answer is not contained within the text below, say 
        "I don't know."\n\nContext:\n"""
        return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

    def answer_query_with_context(
        self,
        query: str,
        df: pd.DataFrame,
        document_embeddings: dict[tuple[str, str], np.array],
        show_prompt: bool = False,
    ) -> str:
        if self.prompt == "":
            self.prompt = self.construct_prompt(query, document_embeddings, df)

        if show_prompt:
            print(self.prompt)

        response = openai.Completion.create(
            prompt=self.prompt, **self.COMPLETIONS_API_PARAMS
        )
        return response["choices"][0]["text"].strip(" \n")


def main():
    load_dotenv()

    api_key = os.environ["OPENAI_API_KEY"]
    kb = KnowledgeBase(api_key)
    df = kb.load_dataset("./data/kyc_labeled.csv")
    document_embeddings = kb.compute_doc_embeddings(df)

    while True:
        # Note: Python 2.x users should use raw_input, the equivalent of 3.x's input
        question = input("Please type your question: ")
        if question == "exit":
            break

        response = kb.answer_query_with_context(question, df, document_embeddings)
        print(colored("\nAnswer: \n\n\t" + response + "\n\n", "green"))


if __name__ == "__main__":
    main()
