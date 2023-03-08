#!/usr/bin/env python3

import os
import nltk
import csv
import pandas as pd
from dotenv import load_dotenv
from atlassian import Confluence
from transformers import GPT2TokenizerFast
from bs4 import BeautifulSoup
from rich.progress import Progress


class Wiki:
    TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")

    def __init__(self):
        load_dotenv()
        self.host = os.environ["CONFLUENCE_HOST"]
        self.username = os.environ["CONFLUENCE_API_USER"]
        self.access_token = os.environ["CONFLUENCE_API_TOKEN"]

    def connect_to_confluence(self) -> Confluence:
        confluence = Confluence(
            url=self.host,
            username=self.username,
            password=self.access_token,
            cloud=True,
        )

        return confluence

    def __read_previously_downloaded(self) -> list[str]:
        downloaded = []
        if os.path.isfile("./data/material.csv") is False:
            return []

        with open("./data/material.csv") as csvfile:
            reader = csv.reader(csvfile)
            num_rows = sum(1 for _ in reader)

        with Progress() as progress:
            task = progress.add_task("[cyan]Retrieve indexed material", total=num_rows)
            progress.update(task, advance=1)
            with open("./data/material.csv") as downloaded_file:
                downloaded_reader = csv.DictReader(downloaded_file)
                for d in downloaded_reader:
                    downloaded.append(d["link"])
                    progress.update(task, advance=1)

        return list(set(downloaded))

    def get_all_pages_from_ids(self, confluence):
        pages = []
        downloaded = self.__read_previously_downloaded()

        with open("./data/source.csv") as csvfile:
            reader = csv.reader(csvfile)
            num_rows = sum(1 for _ in reader)

        with Progress() as progress:
            task = progress.add_task("[cyan]Download missing content", total=num_rows)
            progress.update(task, advance=1)

            with open("./data/source.csv") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    space = row["space"]
                    id = row["page_id"]
                    link = self.host + "/wiki/spaces/" + space + "/pages/" + id
                    if link not in downloaded:
                        page = confluence.get_page_by_id(id, expand="body.storage")
                        pages.append({"space": space, "page": page})

                    progress.update(task, advance=1)

        return pages

    def collect_content_dataframe(self, pages) -> pd.DataFrame:
        collect = []
        for item in pages:
            space = item["space"]
            page = item["page"]
            title = page["title"]
            link = self.host + "/wiki/spaces/" + space + "/pages/" + page["id"]
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

                # Extract the link URLs from the content and append them to the current content
                for sibling in content:
                    if sibling.name == "a":
                        link_url = sibling["href"]
                        current_content += " " + link_url.strip()
                    else:
                        current_content += " " + sibling.get_text().strip()
                # Concatenate the content and add it to the result array
                # for better mapping
                current_content = (
                    header_text
                    + ": "
                    + current_content.replace("\n", " ").replace("\r", " ")
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
                tl = len(tokens)
                if tl >= 100 and tl <= 2046:
                    collect += [(title, link, title + " - " + item, tl)]

        df = pd.DataFrame(collect, columns=["title", "link", "body", "num_tokens"])
        if os.path.isfile("./data/material.csv") is False:
            return df
        else:
            # merge existing files
            old_df = pd.read_csv("./data/material.csv")
            merged_df = pd.concat([old_df, df])

            return merged_df
