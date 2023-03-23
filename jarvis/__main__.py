"""The main entry point. Invoke as `jarvis' or `python -m jarvis'.
"""
import pandas as pd
import numpy as np
import ast
import sys
import time

from jarvis.tokenizer import get_dataframe
from jarvis.downloader import download_content, download_gmail
from jarvis.chat_completion import (
    chat_completion,
    inject_embeddings,
    construct_prompt,
    COMPLETIONS_MODEL,
    ADVANCED_MODEL,
)
from jarvis.status import ExitStatus
from jarvis.constants import MATERIAL_FILE

from collections import deque

USE_GPT_4 = "(gpt-4)"

_last_response = deque(maxlen=3)
_question_history = deque(maxlen=20)


def _query(
    query: str,
    df: pd.DataFrame,
    show_prompt: bool = False,
    show_similarity: bool = False,
):
    prompt, links, similarities, attachments = construct_prompt(query, df)
    prompt = "\n".join(_last_response) + prompt
    deduped_links = list(set(links))

    if USE_GPT_4 in query or len(prompt) + len(query) > 2048:
        model = ADVANCED_MODEL
        max_tokens = 2048
    else:
        model = COMPLETIONS_MODEL
        max_tokens = 1024

    output = _openai_call(prompt, query, model=model, max_tokens=max_tokens)
    _last_response.append(output)
    _question_history.append(query)

    return output, deduped_links, attachments


def _openai_call(prompt, query, model=COMPLETIONS_MODEL, max_tokens=1024) -> str:
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            response = chat_completion(
                prompt, query, model=model, max_tokens=max_tokens
            )
            output = response["choices"][0]["message"]["content"].strip(" \n")

            return output
        except Exception as e:
            retries += 1
            # If the connection is reset, wait for 5 seconds and retry
            time.sleep(5)

    return ""


def _truncate_text(text):
    max_length = 8192 - 2048
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length]


def _reload_csv() -> pd.DataFrame:
    df = pd.read_csv(MATERIAL_FILE)
    df["embeddings"] = df["embeddings"].apply(lambda x: np.array(ast.literal_eval(x)))
    # Safely convert the 'attachments' column from string to list
    df["attachments"] = df["attachments"].apply(lambda x: ast.literal_eval(x))

    return df


def get_latest_gmail() -> list:
    response = []
    gmail_unread = download_gmail()
    if len(gmail_unread) > 0:
        for m in gmail_unread:
            output = _openai_call(
                _truncate_text(m["body"]),
                "Help to condense the email context with subject and summary, please not losing critical details",
                model=ADVANCED_MODEL,
                max_tokens=2048,
            )
            response.append(
                {
                    "link": m["link"],
                    "summary": output,
                }
            )

    return response


def main():
    try:
        pages = download_content(with_gdrive=True, with_confluence=True)
        df = get_dataframe(pages)
        df_with_embedding = inject_embeddings(df)
        df_with_embedding.to_csv(MATERIAL_FILE, index=False)

        # Reload CSV once to prevent formatting misalignment
        final_df = _reload_csv()

        # Listening input from user

    except KeyboardInterrupt:
        exit_status = ExitStatus.ERROR_CTRL_C

    return exit_status.value


if __name__ == "__main__":  # pragma: nocover
    sys.exit(main())
