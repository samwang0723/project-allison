import os
import spacy
import re
import pandas as pd
from transformers import GPT2TokenizerFast
from bs4 import BeautifulSoup, NavigableString
from project_allison.constants import MATERIAL_FILE

TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
MAX_TOKENS = 2046
NLP = spacy.load("en_core_web_sm")
SEPARATOR_DOT = ". "
SEPARATOR = " "


def extract_content(chunk):
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
            title = item["title"] + ": " + match.group(1)
            content = match.group(2)
            soup = BeautifulSoup(content, "html.parser")
            body = parse_html(title, soup)
        elif space == "WEB":
            page = item["page"]
            link = item["link"]
            title_regex = re.compile(r"<title>(.*?)<\/title>")
            title_match = title_regex.search(page)

            if title_match:
                title = title_match.group(1)
            else:
                title = ""
            soup = BeautifulSoup(page, "html.parser")
            body = parse_web(item["title"] + ": " + title, soup)
        elif space == "PDF":
            page = item["page"]
            link = item["link"]
            title = item["title"]
            soup = BeautifulSoup(page, "html.parser")
            body = parse_pdf(title, soup)
        else:
            page = item["page"]
            title = item["title"] + ": " + page["title"]
            link = item["link"]
            htmlbody = page["body"]["storage"]["value"]
            soup = BeautifulSoup(htmlbody, "html.parser")
            body = parse_html(title, soup)

        sum_tokens = 0
        content = []
        for item in body:
            # Calculate number of tokens
            tl = len(TOKENIZER.tokenize(item))

            while tl > 0:
                remaining_space = MAX_TOKENS - sum_tokens

                if tl <= remaining_space:
                    sum_tokens += tl
                    content.append(item)
                    tl = 0
                else:
                    if sum_tokens > 0:
                        collect += _map_reduce(
                            title, link, content, sum_tokens, attachments
                        )
                        content = []
                        sum_tokens = 0

                    # Split the oversized paragraph and process it in smaller parts
                    while tl > MAX_TOKENS:
                        tokens = TOKENIZER.tokenize(item)
                        last_period_index = _find_last_period(tokens, MAX_TOKENS)

                        if (
                            last_period_index == -1
                        ):  # No period found, fallback to cutting at max tokens
                            last_period_index = MAX_TOKENS - 1

                        item_part = TOKENIZER.convert_tokens_to_string(
                            tokens[: last_period_index + 1]
                        )
                        collect += _map_reduce(
                            title,
                            link,
                            [item_part],
                            len(TOKENIZER.tokenize(item_part)),
                            attachments,
                        )

                        # Update the item to include only the unprocessed part
                        item = TOKENIZER.convert_tokens_to_string(
                            tokens[last_period_index + 1 :]
                        )
                        tl = len(TOKENIZER.tokenize(item))

                    # Add the remaining part of the item to content
                    sum_tokens += tl
                    content.append(item)
                    tl = 0

        if len(content) > 0 and sum_tokens > 0:
            collect += _map_reduce(title, link, content, sum_tokens, attachments)

    return collect


def parse_pdf(title, soup) -> list[str]:
    body = []
    body_tags = soup.find_all(["span"])
    current_content = ""
    for p in body_tags:
        current_content += p.get_text(separator=SEPARATOR_DOT)

    body.append(
        title + ": " + current_content.replace("\n", SEPARATOR).replace("\r", SEPARATOR)
    )

    return body


def parse_web(title, soup) -> list[str]:
    body = []
    body_tags = soup.find_all(["p"])
    current_content = ""
    for p in body_tags:
        current_content += p.get_text(separator=SEPARATOR_DOT)

    body.append(
        title + ": " + current_content.replace("\n", SEPARATOR).replace("\r", SEPARATOR)
    )

    return body


def parse_html(title, soup) -> list[str]:
    body = []
    header_tags = soup.find_all(["h1", "h2", "h3"])
    if len(header_tags) == 0:
        # Extract content from paragraph tags if no header tags are found
        paragraph_tags = soup.find_all("p")
        for p in paragraph_tags:
            body.append(
                title
                + ": "
                + p.get_text(separator=SEPARATOR_DOT)
                .replace("\n", SEPARATOR)
                .replace("\r", SEPARATOR)
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
                    if sibling.name == "a" and "href" in sibling.attrs:
                        link_url = sibling["href"]
                        current_content += SEPARATOR + link_url.strip()
                    elif isinstance(sibling, NavigableString):
                        current_content += SEPARATOR + sibling.strip()
                    elif sibling.name == "table":
                        regular_table = _extract_table(sibling, "table")
                        if len(regular_table) > 0:
                            current_content += SEPARATOR + regular_table
                    else:
                        current_content += (
                            SEPARATOR
                            + sibling.get_text(separator=SEPARATOR_DOT).strip()
                        )
                except Exception as e:
                    print(f"[parse_html] Error: {e}")
                    continue
            # Concatenate the content and add it to the result array
            # for better mapping
            current_content = (
                header_text
                + ": "
                + current_content.replace("\n", SEPARATOR).replace("\r", SEPARATOR)
            )

            body.append(current_content)
            current_content = ""

    return body


def get_dataframe(pages) -> pd.DataFrame:
    results = []
    resp = extract_content(pages)
    results.append(resp)

    collect = [
        result for sublist in results for result in sublist
    ]  # combine the results into a single list

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


def _map_reduce(title, link, content, sum_tokens, attachments):
    joined_content = SEPARATOR.join(content)
    return [
        (
            title,
            link,
            title + " - " + joined_content,
            sum_tokens,
            attachments,
        )
    ]


def _find_last_period(tokens, max_tokens):
    last_period_index = -1
    for i, token in enumerate(tokens[:max_tokens]):
        if token == ".":
            last_period_index = i
    return last_period_index


def _extract_table(table, key) -> str:
    if table.name != key:
        return ""

    headers = []
    for header in table.find_all("th"):
        headers.append(header.text)

    text = ""
    for row in table.find_all("tr"):
        row_text = ""
        for i, cell in enumerate(row.find_all("td")):
            if i < len(headers):
                row_text += "(" + headers[i] + "): " + cell.text + " - "
            else:
                row_text += cell.text + " - "
        if row_text:
            text += row_text[:-2] + " | "

    return text
