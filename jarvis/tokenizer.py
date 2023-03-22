import os
import spacy
import re
import multiprocessing
import pandas as pd
from transformers import GPT2TokenizerFast
from bs4 import BeautifulSoup, NavigableString
from functools import partial
from .constants import MATERIAL_FILE

TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
MIN_TOKENS = 30
MAX_TOKENS = 2046
NLP = spacy.load("en_core_web_sm")


class Parser:
    @staticmethod
    def extract_content(chunk, nlp, tokenizer, min_tokens, max_tokens):
        collect = []
        for item in chunk:
            space = item["space"]
            if space == "GOOGLE":
                page = item["page"]
                link = item["link"]

                delimiter1 = "\[title\]"
                delimiter2 = "\[/\]"
                pattern = f"{delimiter1}(.*?){delimiter2}(.*)"

                # Search for the pattern in the text using the re.search() method
                match = re.search(pattern, page)

                # Get the text between the delimiters and the text after the second delimiter
                title = match.group(1)
                content = match.group(2)
                soup = BeautifulSoup(content, "html.parser")
                body = Parser.parse_html(soup, nlp)
            else:
                page = item["page"]
                title = page["title"]
                link = item["link"]
                htmlbody = page["body"]["storage"]["value"]
                soup = BeautifulSoup(htmlbody, "html.parser")
                body = Parser.parse_html(soup, nlp)

            for item in body:
                # Calculate number of tokens
                tl = len(tokenizer.tokenize(item))
                if tl >= min_tokens:  # and tl <= max_tokens:
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
            content = []

            for sibling in siblings:
                if sibling.name in ["h1", "h2", "h3"]:
                    break
                content.append(sibling)

            # Extract the link URLs from the content and append them to the current content
            for sibling in content:
                try:
                    if sibling.name == "a":
                        link_url = sibling["href"]
                        current_content += " " + link_url.strip()
                    elif isinstance(sibling, NavigableString):
                        current_content += " " + sibling.strip()
                    else:
                        current_content += " " + sibling.get_text().strip()
                except:
                    continue
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


def get_dataframe(pages) -> pd.DataFrame:
    num_processes = multiprocessing.cpu_count()  # get the number of available CPUs
    # create a pool of processes
    pool = multiprocessing.Pool(num_processes)

    chunk_size = len(pages) // num_processes
    if chunk_size == 0:
        chunk_size = 1
    chunks = [pages[i : i + chunk_size] for i in range(0, len(pages), chunk_size)]

    tokenizer = TOKENIZER  # initialize your tokenizer object
    min_tokens = MIN_TOKENS  # set your minimum number of tokens
    max_tokens = MAX_TOKENS  # set your maximum number of tokens
    nlp = NLP  # initialize your spacy model
    results = pool.map(
        partial(
            Parser.extract_content,
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

    return __merge_old_content(df)


def __merge_old_content(df):
    if os.path.isfile(MATERIAL_FILE) is False:
        return df
    else:
        # merge existing files
        old_df = pd.read_csv(MATERIAL_FILE)
        merged_df = pd.concat([old_df, df])

        return merged_df
