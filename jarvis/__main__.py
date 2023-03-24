"""The main entry point. Invoke as `jarvis' or `python -m jarvis'.
"""
import pandas as pd
import numpy as np
import ast
import sys
import time
import eventlet

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
from jarvis.constants import MATERIAL_FILE, TEMPLATE_FOLDER, STATIC_FOLDER

from collections import deque
from flask import Flask, render_template
from flask_socketio import SocketIO, send

USE_GPT_4 = "(gpt-4)"

eventlet.monkey_patch()
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
socketio = SocketIO(app)

_last_response = deque(maxlen=3)
_cached_df = deque(maxlen=1)


def _query(
    query: str,
    show_prompt: bool = False,
    show_similarity: bool = False,
):
    prompt, links, similarities, attachments = construct_prompt(query, _cached_df[0])
    prompt = "\n".join(_last_response) + prompt
    deduped_links = list(set(links))

    if show_prompt:
        print("Prompt:\n\t" + prompt)

    if show_similarity:
        print(f"Similarities:\n\t {similarities}")

    if USE_GPT_4 in query or len(prompt) + len(query) > 2048:
        model = ADVANCED_MODEL
        max_tokens = 2048
    else:
        model = COMPLETIONS_MODEL
        max_tokens = 1024

    output = _openai_call(prompt, query, model=model, max_tokens=max_tokens)
    _last_response.append(output)

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
            print(f"Error: {e}, retrying in 5 seconds")
            time.sleep(5)

    return ""


def _truncate_text(text):
    max_length = 8192 - 2048
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length]


def _reload_csv():
    df = pd.read_csv(MATERIAL_FILE)
    df["embeddings"] = df["embeddings"].apply(lambda x: np.array(ast.literal_eval(x)))
    # Safely convert the 'attachments' column from string to list
    df["attachments"] = df["attachments"].apply(lambda x: ast.literal_eval(x))

    _cached_df.append(df)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("message")
def handle_message(message):
    print("Received message: " + message)
    if message == "gmail":
        gmail_unread = download_gmail()
        if len(gmail_unread) > 0:
            send(f"You have {len(gmail_unread)} unread emails")
            for m in gmail_unread:
                output = _openai_call(
                    _truncate_text(m["body"]),
                    "Help to condense the email context with subject and summary, please not losing critical details",
                    model=ADVANCED_MODEL,
                    max_tokens=2048,
                )
                send(output)
        else:
            send("No unread emails.")
    else:
        # Process the message and generate a response (you can use your Python function here)
        response, links, attachments = _query(message)
        send(response)


def main():
    pages = download_content(with_gdrive=True, with_confluence=True)
    df = get_dataframe(pages)
    df_with_embedding = inject_embeddings(df)
    df_with_embedding.to_csv(MATERIAL_FILE, index=False)
    # Reload CSV once to prevent formatting misalignment
    _reload_csv()
    # Listening input from user
    socketio.run(app, port=8000, debug=True)

    return ExitStatus.ERROR_CTRL_C.value


if __name__ == "__main__":  # pragma: nocover
    sys.exit(main())
