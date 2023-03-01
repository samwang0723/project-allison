#!/usr/bin/env python3

import os
import openai
import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from termcolor import colored
from confluence import Wiki
from transformers import GPT2TokenizerFast


class KnowledgeBase:
    COMPLETIONS_MODEL = "text-davinci-003"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    DOC_MODEL = "text-search-curie-doc-001"
    COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": 0.0,
        "max_tokens": 300,
        "model": COMPLETIONS_MODEL,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    MAX_SECTION_LEN = 2046
    SEPARATOR = "\n* "
    ENCODING = "gpt2"  # encoding for text-davinci-003
    TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
    MAX_NUM_TOKENS = 2046

    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.prompt = ""

    def answer_query_with_context(
        self,
        query: str,
        document_embeddings: dict[tuple[str, str], np.array],
        show_prompt: bool = False,
    ) -> str:
        embeddings = self.__order_document_sections_by_query_similarity(
            query, document_embeddings
        )
        prompt, links = self.__construct_prompt(query, embeddings)

        if show_prompt:
            print(prompt)

        response = openai.Completion.create(
            prompt=prompt, **self.COMPLETIONS_API_PARAMS
        )

        output = response["choices"][0]["text"].strip(" \n")

        return output, links

    def decorate_df_with_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df.num_tokens <= self.MAX_NUM_TOKENS]
        df["embeddings"] = df.body.apply(
            lambda x: self.__get_embeddings(x, self.DOC_MODEL)
        )
        return df

    def __get_embeddings(self, text: str, model: str = EMBEDDING_MODEL) -> list[float]:
        result = openai.Embedding.create(model=model, input=text)
        return result["data"][0]["embedding"]

    def __vector_similarity(self, x: list[float], y: list[float]) -> float:
        xx = x[: len(y)]
        return np.dot(np.array(xx), np.array(y))

    def __order_document_sections_by_query_similarity(
        self, query: str, doc_embeddings: pd.DataFrame
    ):
        query_embedding = self.__get_embeddings(query)
        doc_embeddings["similarity"] = doc_embeddings["embeddings"].apply(
            lambda x: self.__vector_similarity(x, query_embedding)
        )
        doc_embeddings.sort_values(by="similarity", inplace=True, ascending=False)
        doc_embeddings.reset_index(drop=True, inplace=True)

        return doc_embeddings

    def __construct_prompt(self, query, doc_embeddings):
        separator_len = len(self.TOKENIZER.tokenize(self.SEPARATOR))

        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_links = []

        for section_index in range(len(doc_embeddings)):
            # Add contexts until we run out of space.
            document_section = doc_embeddings.loc[section_index]
            if document_section.num_tokens <= 200:
                continue

            chosen_sections_len += document_section.num_tokens + separator_len
            if chosen_sections_len > 3000:
                break

            chosen_sections.append(
                self.SEPARATOR + document_section.body.replace("\n", " ")
            )
            chosen_sections_links.append(document_section.link)

        header = """Answer the question as truthfully as possible using the provided 
        context, and if the answer is not contained within the text below, say 
        "I don't know."\n\nContext:\n"""
        prompt = header + "".join(chosen_sections) + "\n\n Q: " + query + "\n A:"

        return (prompt, chosen_sections_links)


def parse_numbers(s):
    return [float(x) for x in s.strip("[]").split(",")]


def get_confluence_embeddings(kb: KnowledgeBase) -> pd.DataFrame:
    # Today's date
    today = datetime.datetime.today()
    # Current file where the embeddings of our internal Confluence document is saved
    Confluence_embeddings_file = "./data/material.csv"
    # Run the embeddings again if the file is more than a week old
    # Otherwise, read the save file
    Confluence_embeddings_file_date = datetime.datetime.fromtimestamp(
        os.path.getmtime(Confluence_embeddings_file)
    )
    delta = today - Confluence_embeddings_file_date
    if delta.days > 7:
        DOC_title_content_embeddings = update_internal_doc_embeddings(kb)
    else:
        DOC_title_content_embeddings = pd.read_csv(
            Confluence_embeddings_file, dtype={"embeddings": object}
        )
        DOC_title_content_embeddings["embeddings"] = DOC_title_content_embeddings[
            "embeddings"
        ].apply(lambda x: parse_numbers(x))

    return DOC_title_content_embeddings


def update_internal_doc_embeddings(kb: KnowledgeBase) -> pd.DataFrame:
    wiki = Wiki()
    confluence = wiki.connect_to_confluence()
    pages = wiki.get_all_pages(confluence)
    df = wiki.collect_content_dataframe(pages)
    df = kb.decorate_df_with_embeddings(df)
    df.to_csv("./data/material.csv", index=False)

    return df


def main():
    load_dotenv()

    kb = KnowledgeBase()
    df = get_confluence_embeddings(kb)

    while True:
        # Note: Python 2.x users should use raw_input, the equivalent of 3.x's input
        question = input("Please type your question: ")
        if question == "exit":
            break

        response, links = kb.answer_query_with_context(question, df)
        print(
            colored(
                "\nAnswer: \n\n\t"
                + response
                + "\n\nLinks: \n\n\t"
                + str(links)
                + "\n\n",
                "green",
            )
        )


if __name__ == "__main__":
    main()
