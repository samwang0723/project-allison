"""The main entry point. Invoke as `jarvis' or `python -m jarvis'.
"""
import pandas as pd
import numpy as np
import ast
import sys
import time
import eventlet
import os
import base64
import uuid
import io

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
from jarvis.constants import ENV_PATH

from dotenv import load_dotenv
from collections import deque
from flask import Flask, render_template, session
from flask_socketio import SocketIO, send
from PIL import Image

USE_GPT_4 = "(gpt-4)"
HELP_TEXT = """
1. command:fetch_gmail
2. command:similarity
3. command:prompt
4. command:reload_csv
5. command:save:{file_name}
6. command:diagram:
7. command:reset_session
"""
MAX_HISTORY = 3
STOP_SIGN = "[[stop]]"

load_dotenv(dotenv_path=ENV_PATH)
eventlet.monkey_patch()
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.secret_key = os.environ["FLASK_SECRET_KEY"]
app.config["PERMANENT_SESSION_LIFETIME"] = 1800  # 30 minutes in seconds
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
socketio = SocketIO(app)
_global_df_cache = deque(maxlen=1)


def _query(query: str):
    history_records = session.get("history", None)
    if history_records is not None:
        prompt, links, similarities, attachments = construct_prompt(
            query, _global_df_cache[0]
        )
        prompt = "\n".join(history_records) + prompt
        deduped_links = list(set(links))

        if USE_GPT_4 in query or len(prompt) + len(query) > 4096:
            model = ADVANCED_MODEL
            max_tokens = 2048
        else:
            model = COMPLETIONS_MODEL
            max_tokens = 1024

        output = _openai_call(prompt, query, model=model, max_tokens=max_tokens)
        if len(history_records) >= MAX_HISTORY:
            history_records.pop(0)
        history_records.append(output)

    return output, deduped_links, attachments, prompt, similarities


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
    print("[4] Reloading from CSV sources")
    _global_df_cache.clear()

    df = pd.read_csv(MATERIAL_FILE)
    df["embeddings"] = df["embeddings"].apply(lambda x: np.array(ast.literal_eval(x)))
    # Safely convert the 'attachments' column from string to list
    df["attachments"] = df["attachments"].apply(lambda x: ast.literal_eval(x))

    _global_df_cache.append(df)


def _handle_command(message):
    if "fetch_gmail" in message:
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
    elif "reset_session" in message:
        if session.get("history", []):
            session["history"].clear()
        send("Session being reset successfully.")
    elif "reload_csv" in message:
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
    elif "similarity" in message:
        if session.get("similarity", False) == False:
            session["similarity"] = True
            send("Similarity scores will be shown.")
        else:
            session["similarity"] = False
            send("Similarity scores will be hidden.")
    elif "prompt" in message:
        if session.get("prompt", False) == False:
            session["prompt"] = True
            send("Prompt will be shown.")
        else:
            session["prompt"] = False
            send("Prompt will be hidden.")
    elif "help:" in message:
        send("```" + HELP_TEXT + "```")
    else:
        send("Command not found. Please try again.")


@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


@socketio.on("upload_image")
def handle_upload_image(data):
    image_data = data["image"]
    filename = data["filename"]
    # Extract the file extension from the filename
    file_ext = os.path.splitext(filename)[1]
    # Generate a unique filename to avoid overwriting existing files
    unique_filename = str(uuid.uuid4()) + file_ext
    full_path = f"{STATIC_FOLDER}/tmp/{unique_filename}"
    # Save the image with the original filename
    with open(full_path, "wb") as f:
        f.write(base64.b64decode(image_data.split(",")[1]))
    print(f"Image saved as {full_path}")

    send(
        f"Image received [{unique_filename}](/static/tmp/{unique_filename}), unfortunately I can't do anything with it yet :("
    )
    send("[[stop]]")


@socketio.on("message")
def handle_message(message):
    try:
        history_records = session.get("history", None)
        if history_records is None:
            session["history"] = []
            session["similarity"] = False
            session["prompt"] = False

        if "command:" in message:
            _handle_command(message)
        else:
            # Process the message and generate a response (you can use your Python function here)
            _, links, attachments, prompt, similarities = _query(message)
            output = ""
            if len(links) > 0:
                output += "\n\nSources:\n"
            for link in links:
                output += f"\t{link}\n"
            if len(attachments) > 0:
                output += "\n\nAttachments:\n"
            for attachment in attachments:
                output += f"\t{attachment}\n"
            if session.get("prompt", False):
                output += f"\n\nPrompt:\n\t{prompt}"
            if session.get("similarity", False):
                similarities = ", ".join(similarities)
                output += f"\n\nSimilarity:\n\t{similarities}"

            send(output)

        send(STOP_SIGN)
    except Exception as e:
        send(f"Error occurred: {e}")


def main():
    print("[1] Downloading content from Google Drive and Confluence")
    pages = download_content(with_gdrive=True, with_confluence=True)
    df = get_dataframe(pages)
    print("[2] Calculate embeddings based on dataframe")
    df_with_embedding = inject_embeddings(df)
    print("[3] Saving indexed CSV file")
    df_with_embedding.to_csv(MATERIAL_FILE, index=False)
    # Reload CSV once to prevent formatting misalignment
    _reload_csv()
    # Listening input from user
    socketio.run(app, host="0.0.0.0", port=8000, debug=True)

    return ExitStatus.ERROR_CTRL_C.value


if __name__ == "__main__":  # pragma: nocover
    sys.exit(main())
