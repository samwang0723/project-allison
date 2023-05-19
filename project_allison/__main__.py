import select
import sys
import eventlet
import os
import base64
import uuid
import inspect

from project_allison.commands import handle_system_command, handle_tasks
from project_allison.tokenizer import get_dataframe
from project_allison.downloader import (
    download_content,
    load_plugins,
)
from project_allison.chat_completion import (
    openai_call,
    construct_prompt,
    COMPLETIONS_MODEL,
    ADVANCED_MODEL,
    parse_task_prompt,
)
from project_allison.status import ExitStatus
from project_allison.constants import TEMPLATE_FOLDER, STATIC_FOLDER
from project_allison.constants import ENV_PATH
from project_allison.vectordb import get_vector_collection, insert_vendor_knowledgebase

from dotenv import load_dotenv
from flask import Flask, render_template, session
from flask_socketio import SocketIO, send
from collections import deque


USE_GPT_4 = "(gpt-4)"
MAX_HISTORY = 3
STOP_SIGN = "[[stop]]"

load_dotenv(dotenv_path=ENV_PATH)
eventlet.monkey_patch()
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.secret_key = os.environ["FLASK_SECRET_KEY"]
app.config["PERMANENT_SESSION_LIFETIME"] = 1800  # 30 minutes in seconds
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
socketio = SocketIO(app)
_global_collection = deque(maxlen=1)


def _query(query: str):
    history_records = session.get("history", None)
    if history_records is not None:
        deduped_links, attachments, prompt, similarities = [], [], "", []
        # identify task commands
        if query.startswith("/"):
            task_data = parse_task_prompt(query[1:], "\n".join(history_records))
            handle_tasks(task_data)
            output = "Task completed"
        else:  # Regular conversational call
            prompt, links, similarities, attachments = construct_prompt(
                query, _global_collection[0]
            )
            prompt = "\n".join(history_records) + prompt
            deduped_links = list(set(links))

            if USE_GPT_4 in query or len(prompt) + len(query) > 4096:
                model = ADVANCED_MODEL
                max_tokens = 2048
            else:
                model = COMPLETIONS_MODEL
                max_tokens = 1024

            output = openai_call(prompt, query, model=model, max_tokens=max_tokens)
            if len(history_records) >= MAX_HISTORY:
                history_records.pop(0)
            history_records.append(f"Question: {query}. Answer: {output}. ")

    return output, deduped_links, attachments, prompt, similarities


@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


@socketio.on("upload_image")
def handle_upload_image(data):
    try:
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
    except Exception as e:
        print(f"{inspect.currentframe().f_code.co_name}, Error: {e}")
        send("Error: " + str(e))


@socketio.on("mode")
def handle_mode(mode):
    try:
        if mode == "desktop":
            send("Switching to desktop mode...")
            session["mode"] = "desktop"
        else:
            send("Switching to mobile mode...")
            session["mode"] = "mobile"
    except Exception as e:
        print(f"{inspect.currentframe().f_code.co_name}, Error: {e}")
        send("Error: " + str(e))

    send(STOP_SIGN)


@socketio.on("message")
def handle_message(message):
    try:
        print(f"Received message: {message}")
        history_records = session.get("history", None)
        if history_records is None:
            session["history"] = []
            session["similarity"] = False
            session["prompt"] = False

        if message.startswith("command:"):
            action = message.split("command:")[1].strip()
            handle_system_command(action)
        else:
            # Process the message and generate a response (you can use your Python function here)
            _, links, attachments, prompt, similarities = _query(message)
            output = ""
            if len(links) > 0:
                output += "\n\nSources:\n"
                for link in links:
                    output += f"\t{link}\n"

            if session.get("prompt", False):
                output += f"\n\nPrompt:\n\t{prompt}"

            if session.get("similarity", False):
                similarities = ", ".join(similarities)
                output += f"\n\nSimilarity:\n\t{similarities}"

            mode = session.get("mode", None)
            if len(attachments) > 0 and mode == "desktop":
                output += "\n\nAttachments:\n"

                for attachment in attachments:
                    output += f"\t{attachment}\n"

            send(output)
    except Exception as e:
        send(f"{inspect.currentframe().f_code.co_name}, Error: {e}")

    send(STOP_SIGN)


def main():
    load_plugins()

    _global_collection.append(get_vector_collection("project-allison"))
    print(
        f"[1] Downloading content from external sources, collection={_global_collection[0]}"
    )

    pages = download_content("source", vdb=_global_collection[0])
    df = get_dataframe(pages)

    print("[2] Index into vector database")
    insert_vendor_knowledgebase(_global_collection[0], df)

    # Listening input from user
    socketio.run(app, host="0.0.0.0", port=8009, debug=True, use_reloader=False)

    return ExitStatus.ERROR_CTRL_C.value


if __name__ == "__main__":  # pragma: nocover
    sys.exit(main())
