#!/usr/bin/env python3

import os
import nltk
import pandas as pd
from dotenv import load_dotenv
from atlassian import Confluence
from transformers import GPT2TokenizerFast
from bs4 import BeautifulSoup


class Wiki:
    URL = "https://mcoproduct.atlassian.net"
    CRAWL_LIMIT = 100
    TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
    MAX_NUM_TOKENS = 2046

    def __init__(self):
        load_dotenv()

        self.username = os.environ["CONFLUENCE_API_USER"]
        self.access_token = os.environ["CONFLUENCE_API_TOKEN"]

    def connect_to_confluence(self) -> Confluence:
        confluence = Confluence(
            url=self.URL, username=self.username, password=self.access_token, cloud=True
        )

        return confluence

    def get_all_pages(self, confluence, space="COM"):
        # There is a limit of how many pages we can retrieve one at a time
        # so we retrieve 100 at a time and loop until we know we retrieved all of
        # them.
        keep_going = True
        start = 0
        pages = []
        while keep_going:
            results = confluence.get_all_pages_from_space(
                space,
                start=start,
                limit=self.CRAWL_LIMIT,
                status=None,
                expand="body.storage",
                content_type="page",
            )
            pages.extend(results)
            if len(results) < self.CRAWL_LIMIT:
                keep_going = False
            else:
                start = start + self.CRAWL_LIMIT

        return pages

    def collect_content_dataframe(self, pages) -> pd.DataFrame:
        collect = []
        for page in pages:
            title = page["title"]
            link = self.URL + "/wiki/spaces/COM/pages/" + page["id"]
            htmlbody = page["body"]["storage"]["value"]
            htmlParse = BeautifulSoup(htmlbody, "html.parser")
            body = []
            for para in htmlParse.find_all("p"):
                # Keep only a sentence if there is a subject and a verb
                # Otherwise, we assume the sentence does not contain enough useful information
                # to be included in the context for openai
                sentence = para.get_text()
                tokens = nltk.tokenize.word_tokenize(sentence)
                token_tags = nltk.pos_tag(tokens)
                tags = [x[1] for x in token_tags]
                if any([x[:2] == "VB" for x in tags]):  # There is at least one verb
                    if any([x[:2] == "NN" for x in tags]):  # There is at least noun
                        body.append(sentence)
            body = ". ".join(body)
            # Calculate number of tokens
            tokens = self.TOKENIZER.encode(body)
            collect += [(title, link, body, len(tokens))]

        df = pd.DataFrame(collect, columns=["title", "link", "body", "num_tokens"])
        # Calculate the embeddings
        # Limit first to pages with less than 2046 tokens
        df = df[df.num_tokens <= self.MAX_NUM_TOKENS]

        return df
