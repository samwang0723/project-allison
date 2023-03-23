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
MAX_TOKENS = 2046
NLP = spacy.load("en_core_web_sm")


class Parser:
    @staticmethod
    def extract_content(chunk, nlp, tokenizer, max_tokens):
        collect = []
        for item in chunk:
            space = item["space"]
            attachments = item.get("attachments", [])

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
                body = Parser.parse_html(title, soup, nlp)
            else:
                page = item["page"]
                title = page["title"]
                link = item["link"]
                htmlbody = page["body"]["storage"]["value"]
                soup = BeautifulSoup(htmlbody, "html.parser")
                body = Parser.parse_html(title, soup, nlp)

            sum_tokens = 0
            content = []
            for item in body:
                # Calculate number of tokens
                tl = len(tokenizer.tokenize(item))
                if sum_tokens + tl <= max_tokens:
                    sum_tokens += tl
                    content.append(item)
                else:
                    joined_content = " ".join(content)
                    collect += [
                        (
                            title,
                            link,
                            title + " - " + joined_content,
                            sum_tokens,
                            attachments,
                        )
                    ]
                    content = [item]
                    sum_tokens = tl

            if len(content) > 0:
                joined_content = " ".join(content)
                collect += [
                    (
                        title,
                        link,
                        title + " - " + joined_content,
                        sum_tokens,
                        attachments,
                    )
                ]

        return collect

    @staticmethod
    def parse_html(title, soup, nlp) -> list[str]:
        body = []
        header_tags = soup.find_all(["h1", "h2", "h3"])

        if len(header_tags) == 0:
            # Extract content from paragraph tags if no header tags are found
            paragraph_tags = soup.find_all("p")
            for p in paragraph_tags:
                body.append(
                    title + ": " + p.get_text().replace("\n", " ").replace("\r", " ")
                )
        else:
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
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
                # Concatenate the content and add it to the result array
                # for better mapping
                current_content = (
                    header_text
                    + ": "
                    + current_content.replace("\n", " ").replace("\r", " ")
                )

                # Check if the current content contains at least 2 verbs and nouns
                contains_verb_noun = 0
                doc = nlp(current_content)
                for token in doc:
                    if token.pos_ == "VERB":
                        contains_verb_noun += 1
                    elif token.pos_ == "NOUN":
                        contains_verb_noun += 1

                if contains_verb_noun >= 2:
                    body.append(current_content)

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
    max_tokens = MAX_TOKENS  # set your maximum number of tokens
    nlp = NLP  # initialize your spacy model
    results = pool.map(
        partial(
            Parser.extract_content,
            nlp=nlp,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        ),
        chunks,
    )  # process each chunk in a separate process
    collect = [
        result for sublist in results for result in sublist
    ]  # combine the results into a single list

    pool.close()
    pool.join()

    df = pd.DataFrame(
        collect, columns=["title", "link", "body", "num_tokens", "attachments"]
    )

    return __merge_old_content(df)


def __merge_old_content(df):
    if os.path.isfile(MATERIAL_FILE) is False:
        return df
    else:
        # merge existing files
        old_df = pd.read_csv(MATERIAL_FILE)
        merged_df = pd.concat([old_df, df])

        return merged_df
