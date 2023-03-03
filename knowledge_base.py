#!/usr/bin/env python3

import os
import openai
import datetime
import tiktoken
import time
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from termcolor import colored
from confluence import Wiki
from transformers import GPT2TokenizerFast
from collections import deque


class KnowledgeBase:
    COMPLETIONS_MODEL = "gpt-3.5-turbo"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": 0.0,
        "max_tokens": 1024,
        "model": COMPLETIONS_MODEL,
        "top_p": 0.1,
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
        self.encoding = tiktoken.get_encoding(self.ENCODING)
        self.separator_len = len(self.encoding.encode(self.SEPARATOR))
        self.last_response = deque(maxlen=3)
        self.question_history = deque(maxlen=20)

    def answer_query_with_context(
        self,
        query: str,
        df: pd.DataFrame,
        document_embeddings: dict[tuple[str, str], np.array],
        show_prompt: bool = False,
    ) -> str:
        prompt, links = self.__construct_prompt(query, document_embeddings, df)
        prompt = "\n".join(self.last_response) + prompt
        deduped_links = list(set(links))

        if show_prompt:
            print(colored("Prompt:", "red"), prompt)
        try:
            response = openai.ChatCompletion.create(
                **self.COMPLETIONS_API_PARAMS,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
            )
        except openai.errors.APIConnectionError:
            print(colored("Openai connection reset, wait for 5 secs", "red"))
            # If the connection is reset, wait for 5 seconds and retry
            time.sleep(5)
            response = openai.ChatCompletion.create(
                **self.COMPLETIONS_API_PARAMS,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
            )

        output = response["choices"][0]["message"]["content"].strip(" \n")
        self.last_response.append(output)
        self.question_history.append(query)

        return output, deduped_links

    def decorate_df_with_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        df["embeddings"] = df.body.apply(lambda x: self.__get_embeddings(x))
        return df

    def compute_doc_embeddings(
        self, df: pd.DataFrame
    ) -> dict[tuple[str, str], list[float]]:
        return {idx: r["embeddings"] for idx, r in df.iterrows()}

    def __get_embeddings(self, text: str, model: str = EMBEDDING_MODEL) -> list[float]:
        result = openai.Embedding.create(model=model, input=text)
        return result["data"][0]["embedding"]

    def __vector_similarity(self, x: list[float], y: list[float]) -> float:
        return np.dot(np.array(x), np.array(y))

    def __order_document_sections_by_query_similarity(
        self, query: str, contexts: dict[tuple[str, str], np.array]
    ) -> list[tuple[float, tuple[str, str]]]:
        query_embedding = self.__get_embeddings(query)
        document_similarities = sorted(
            [
                (self.__vector_similarity(query_embedding, doc_embedding), doc_index)
                for doc_index, doc_embedding in contexts.items()
            ],
            reverse=True,
        )

        return document_similarities

    def __construct_prompt(
        self, question: str, context_embeddings: dict, df: pd.DataFrame
    ):
        most_relevant_document_sections = (
            self.__order_document_sections_by_query_similarity(
                question, context_embeddings
            )
        )

        chosen_sections = []
        chosen_sections_links = []
        chosen_sections_len = 0

        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.
            df2 = df.sort_index()
            document_section = df2.loc[section_index]

            chosen_sections_len += int(document_section.num_tokens) + self.separator_len
            if chosen_sections_len > self.MAX_SECTION_LEN:
                break

            chosen_sections.append(
                str(self.SEPARATOR + document_section.body.replace("\n", " "))
            )
            chosen_sections_links.append(document_section.link)

        header = """You are a helpful assistant that can answer questions about 
        Crypto.com specific knowledge giving below context.\n\nContext:\n"""
        prompt = header + "".join(chosen_sections)

        return (prompt, chosen_sections_links)


def parse_numbers(s):
    return [float(x) for x in s.strip("[]").split(",")]


def update_internal_doc_embeddings(kb: KnowledgeBase) -> pd.DataFrame:
    print(colored("Updating internal document embeddings from Confluence...\n", "red"))

    wiki = Wiki()
    confluence = wiki.connect_to_confluence()
    pages = wiki.get_all_pages_from_ids(confluence)
    df = wiki.collect_content_dataframe(pages)
    df = kb.decorate_df_with_embeddings(df)
    df.to_csv("./data/material.csv", index=False)

    print(colored("Confluence download and index completed!\n", "yellow"))

    return df


def print_result(response, links):
    print(colored("\nAnswer: ", "green"))

    pattern = re.compile(r"```(.+?)```", re.DOTALL)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches:
            response = response.replace(
                f"```{match}```", colored(f"```{match}```", "yellow")
            )
    print("\t" + response)
    print(colored("\n\nLinks: ", "green"))
    print("\t" + str(links) + "\n\n")


def main():
    load_dotenv()

    kb = KnowledgeBase()
    df = update_internal_doc_embeddings(kb)

    # TODO: This is using runtime computation of embeddings, which is slow.
    # We should save the embeddings to a file and load them from there.
    document_embeddings = kb.compute_doc_embeddings(df)
    prompt_on = False

    while True:
        # Note: Python 2.x users should use raw_input, the equivalent of 3.x's input
        print("--------------------")
        question = input(colored("Please type your question: ", "cyan"))
        if question == "exit":
            break
        elif question == "show-prompt":
            prompt_on = True
            print(colored("Prompt is now on for next conversation", "cyan"))
            continue
        elif question == "hide-prompt":
            prompt_on = False
            print(colored("Prompt is now off for next conversation", "cyan"))
            continue
        elif question == "clear":
            kb.last_response.clear()
            kb.question_history.clear()
            print(colored("Conversation history cleared", "cyan"))
            continue
        elif question == "history":
            print(
                colored("\nConversation history:\n\n\t．", "cyan"),
                "\n\t．".join(kb.question_history),
            )
            continue
        elif question == "help":
            print(
                colored(
                    """
    exit: exit the program
    show-prompt: show prompt for next conversation
    hide-prompt: hide prompt for next conversation
    clear: clear conversation history
    history: show conversation history
    help: show this help message
                    """,
                    "cyan",
                )
            )
            continue

        response, links = kb.answer_query_with_context(
            question, df, document_embeddings, prompt_on
        )
        print_result(response, links)

        prompt_on = False


if __name__ == "__main__":
    main()
