import os
import spacy
import csv
import multiprocessing
import pandas as pd
from dotenv import load_dotenv
from atlassian import Confluence
from transformers import GPT2TokenizerFast
from bs4 import BeautifulSoup
from rich.progress import Progress
from functools import partial


class Wiki:
    TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
    MATERIAL_FILE = "./data/material.csv"
    SOURCE_FILE = "./data/source.csv"
    MIN_TOKENS = 30
    MAX_TOKENS = 2046

    def __init__(self):
        load_dotenv()
        self.host = os.environ["CONFLUENCE_HOST"]
        self.username = os.environ["CONFLUENCE_API_USER"]
        self.access_token = os.environ["CONFLUENCE_API_TOKEN"]
        self.nlp = spacy.load("en_core_web_sm")

    def connect_to_confluence(self) -> Confluence:
        confluence = Confluence(
            url=self.host,
            username=self.username,
            password=self.access_token,
            cloud=True,
        )

        return confluence

    def get_all_pages_from_ids(self, confluence):
        pages = []
        downloaded = self.__read_previously_downloaded()

        with open(self.SOURCE_FILE) as csvfile:
            reader = csv.reader(csvfile)
            num_rows = sum(1 for _ in reader)

        with Progress() as progress:
            task = progress.add_task("[cyan]Download missing content", total=num_rows)
            progress.update(task, advance=1)

            with open(self.SOURCE_FILE) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    space = row["space"]
                    id = row["page_id"]
                    link = self.host + "/wiki/spaces/" + space + "/pages/" + id
                    if link not in downloaded:
                        page = confluence.get_page_by_id(id, expand="body.storage")
                        pages.append({"page": page, "link": link})

                    progress.update(task, advance=1)

        return pages

    def collect_with_processes(self, pages) -> pd.DataFrame:
        num_processes = multiprocessing.cpu_count()  # get the number of available CPUs
        # create a pool of processes
        pool = multiprocessing.Pool(num_processes)

        chunk_size = len(pages) // num_processes
        if chunk_size == 0:
            chunk_size = 1
        chunks = [pages[i : i + chunk_size] for i in range(0, len(pages), chunk_size)]

        tokenizer = self.TOKENIZER  # initialize your tokenizer object
        min_tokens = self.MIN_TOKENS  # set your minimum number of tokens
        max_tokens = self.MAX_TOKENS  # set your maximum number of tokens
        nlp = self.nlp  # initialize your spacy model
        results = pool.map(
            partial(
                Wiki.extract_content,
                nlp=nlp,
                tokenizer=tokenizer,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            ),
            chunks,
        )  # process each chunk in a separate process
        collect = [
            result for sublist in results for result in sublist
        ]  # combine the results into a single list

        pool.close()
        pool.join()

        df = pd.DataFrame(collect, columns=["title", "link", "body", "num_tokens"])

        return self.__merge_old_content(df)

    @staticmethod
    def extract_content(chunk, nlp, tokenizer, min_tokens, max_tokens):
        collect = []
        for item in chunk:
            page = item["page"]
            title = page["title"]
            link = item["link"]
            htmlbody = page["body"]["storage"]["value"]
            soup = BeautifulSoup(htmlbody, "html.parser")
            body = Wiki.parse_html(soup, nlp)

            for item in body:
                # Calculate number of tokens
                tl = len(tokenizer.tokenize(item))
                if tl >= min_tokens and tl <= max_tokens:
                    collect += [(title, link, title + " - " + item, tl)]

        return collect

    @staticmethod
    def parse_html(soup, nlp) -> list[str]:
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

            # Check if the current content contains at least 2 verbs and nouns
            contains_verb_noun = 0
            doc = nlp(current_content)
            for token in doc:
                if token.pos_ == "VERB":
                    contains_verb_noun += 1
                elif token.pos_ == "NOUN":
                    contains_verb_noun += 1

            if contains_verb_noun >= 2:
                body.append(current_content + " ")

            current_content = ""

        return body

    def __merge_old_content(self, df):
        if os.path.isfile(self.MATERIAL_FILE) is False:
            return df
        else:
            # merge existing files
            old_df = pd.read_csv(self.MATERIAL_FILE)
            merged_df = pd.concat([old_df, df])

            return merged_df

    def __read_previously_downloaded(self) -> list[str]:
        downloaded = []
        if os.path.isfile(self.MATERIAL_FILE) is False:
            return []

        with open(self.MATERIAL_FILE) as csvfile:
            reader = csv.reader(csvfile)
            num_rows = sum(1 for _ in reader)

        with Progress() as progress:
            task = progress.add_task("[cyan]Retrieve indexed material", total=num_rows)
            progress.update(task, advance=1)
            with open(self.MATERIAL_FILE) as downloaded_file:
                downloaded_reader = csv.DictReader(downloaded_file)
                for d in downloaded_reader:
                    downloaded.append(d["link"])
                    progress.update(task, advance=1)

        return list(set(downloaded))
