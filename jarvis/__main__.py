"""The main entry point. Invoke as `jarvis' or `python -m jarvis'.
"""
import pandas as pd
import numpy as np
import ast
import sys
import time
import eventlet
import os

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
            # output = response["choices"][0]["message"]["content"].strip(" \n")
            # create variables to collect the stream of chunks
            # iterate through the stream of events
            collected_messages = []
            for chunk in response:
                # extract the message
                chunk_message = chunk["choices"][0]["delta"]
                collected_messages.append(chunk_message)
                # print the delay and text
                if "content" in chunk_message:
                    send(chunk_message["content"])

            full_reply_content = "".join(
                [m.get("content", "") for m in collected_messages]
            )
            return full_reply_content
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
    _cached_df.clear()

    df = pd.read_csv(MATERIAL_FILE)
    df["embeddings"] = df["embeddings"].apply(lambda x: np.array(ast.literal_eval(x)))
    # Safely convert the 'attachments' column from string to list
    df["attachments"] = df["attachments"].apply(lambda x: ast.literal_eval(x))

    _cached_df.append(df)


def _handle_command(message):
    if "gmail" in message:
        gmail_unread = download_gmail()
        if len(gmail_unread) > 0:
            send(f"You have ___`{len(gmail_unread)}`___ unread emails")
            for m in gmail_unread:
                send("\n\n---\n\n")
                _openai_call(
                    _truncate_text(m["body"]),
                    "[DO NOT create a email response] Condense the email context with subject and summary, not losing critical details.",
                    model=ADVANCED_MODEL,
                    max_tokens=2048,
                )
                send("\n\nSources:\n\t" + m["link"])
        else:
            send("No unread emails.")
    elif "reset session" in message:
        _last_response.clear()
        send("Session being reset successfully.")
    elif "reload csv" in message:
        _reload_csv()
        send("CSV sources reloaded.")
    elif "save:" in message:
        # parse the message to get the file name
        lines = message.split("save:")[1].split("\n")
        file_name = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        content = content.replace("```\n", "").replace("\n```", "")
        full_path = f"{STATIC_FOLDER}/tmp/{file_name}"
        with open(full_path, "w") as f:
            f.write(content)
        send(f"File `{full_path}` saved successfully.")
    elif "diagram:" in message:
        # parse the message to get the content
        lines = message.split("diagram:")[1].split("\n")
        file_name = "diagram.dot"
        content = "\n".join(lines[1:]).strip()
        content = content.replace("```\n", "").replace("\n```", "")
        full_path = f"{STATIC_FOLDER}/tmp/{file_name}"
        with open(full_path, "w") as f:
            f.write(content)

        output_path = f"{STATIC_FOLDER}/tmp/diagram.png"
        dot_command = ["dot", "-Tpng", full_path, "-o", output_path]
        os.popen(" ".join(dot_command)).read()
        send("File [diagram.png](static/tmp/diagram.png) saved successfully.")


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("message")
def handle_message(message):
    print("Received message: " + message)
    try:
        if "command:" in message:
            _handle_command(message)
        else:
            # Process the message and generate a response (you can use your Python function here)
            resp, links, attachments = _query(message)
            output = ""
            if len(links) > 0:
                output += "\n\nSources:\n"
            for link in links:
                output += f"\t{link}\n"
            if len(attachments) > 0:
                output += "\n\nAttachments:\n"
            for attachment in attachments:
                output += f"\t{attachment}\n"

            print(resp)
            send(output)

        send("[[stop]]")
    except Exception as e:
        send(f"Error occurred: {e}")


def main():
    pages = download_content(with_gdrive=True, with_confluence=True)
    df = get_dataframe(pages)
    df_with_embedding = inject_embeddings(df)
    df_with_embedding.to_csv(MATERIAL_FILE, index=False)
    # Reload CSV once to prevent formatting misalignment
    _reload_csv()
    # Listening input from user
    socketio.run(app, host="0.0.0.0", port=8000, debug=True)

    return ExitStatus.ERROR_CTRL_C.value


if __name__ == "__main__":  # pragma: nocover
    sys.exit(main())
