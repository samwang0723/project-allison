#!/usr/bin/env python3

import os
import nltk
import csv
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

    def get_all_pages_from_ids(self, confluence):
        pages = []
        with open("./data/source.csv") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                val = row["page_id"]
                page = confluence.get_page_by_id(val, expand="body.storage")
                pages.append(page)
        return pages

    def collect_content_dataframe(self, pages) -> pd.DataFrame:
        collect = []
        for page in pages:
            title = page["title"]
            link = self.URL + "/wiki/spaces/COM/pages/" + page["id"]
            htmlbody = page["body"]["storage"]["value"]
            soup = BeautifulSoup(htmlbody, "html.parser")
            body = []

            header_tags = soup.find_all(["h1", "h2", "h3"])
            current_content = ""

            for i in range(len(header_tags)):
                # Extract the text from the current header tag
                header_text = header_tags[i].get_text().strip()

                # Extract the text of all the siblings until the next header tag
                siblings = header_tags[i].next_siblings
                content = [
                    sibling
                    for sibling in siblings
                    if sibling.name != "h1"
                    and sibling.name != "h2"
                    and sibling.name != "h3"
                ]

                # Concatenate the content and add it to the result array
                current_content = (
                    header_text
                    + ": "
                    + " ".join([c.get_text().strip() for c in content])
                )

                if "changelog" in current_content.lower():
                    continue

                tokens = nltk.tokenize.word_tokenize(current_content)
                token_tags = nltk.pos_tag(tokens)
                tags = [x[1] for x in token_tags]
                if any([x[:2] == "VB" for x in tags]):  # There is at least one verb
                    if any([x[:2] == "NN" for x in tags]):  # There is at least noun
                        body.append(current_content + " ")

            for item in body:
                # Calculate number of tokens
                tokens = self.TOKENIZER.encode(item)
                if len(tokens) >= 100:
                    collect += [(title, link, title + " - " + item, len(tokens))]

        df = pd.DataFrame(collect, columns=["title", "link", "body", "num_tokens"])
        # Calculate the embeddings
        # Limit first to pages with less than 2046 tokens
        df = df[df.num_tokens <= self.MAX_NUM_TOKENS]

        return df
