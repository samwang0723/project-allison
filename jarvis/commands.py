import os

from jarvis.constants import STATIC_FOLDER
from jarvis.downloader import download_content
from jarvis.chat_completion import openai_call, ADVANCED_MODEL

from flask import session
from flask_socketio import send
from texttable import Texttable

_system_commands = ["similarity", "prompt", "reset_session"]
_file_operation_commands = ["save:", "diagram:"]
_business_logic_commands = ["fetch_gmail", "fetch_news", "fetch_finance"]


def handle_command(action):
    if action in _system_commands:
        _handle_system_command(action)
    elif action in _file_operation_commands:
        _handle_file_operation_command(action)
    elif action in _business_logic_commands:
        _handle_business_logic_command(action)
    else:
        send("Command not found. Please try again.")


def _handle_business_logic_command(message):
    if "fetch_gmail" in message:
        gmail_unread = download_content("gmail")
        if len(gmail_unread) > 0:
            send(f"You have ___`{len(gmail_unread)}`___ unread emails")
            for m in gmail_unread:
                send("\n\n---\n\n")
                openai_call(
                    _truncate_text(m["body"]),
                    "[DO NOT CREATE RESPONSE] Condense email context with subject and summary, don't lose date,item,person,numbers, etc.",
                    model=ADVANCED_MODEL,
                    max_tokens=2048,
                )
                send("\n\nSources:\n\t" + m["link"])
        else:
            send("No unread emails.")
    elif "fetch_news" in message:
        news = download_content("news")
        if len(news) > 0:
            send(f"Found ___`{len(news)}`___ news updates")
            for m in news:
                send("\n\n---\n\n")
                openai_call(
                    _truncate_text(m["body"]),
                    "Condense the news context with subject and summary, not losing critical details.",
                    model=ADVANCED_MODEL,
                    max_tokens=2048,
                )
                send("\n\nSources:\n\t" + m["link"])
        else:
            send("No unread news.")
    elif "fetch_finance" in message:
        picked_stocks = download_content("finance::picked")
        if len(picked_stocks) > 0:
            print(picked_stocks)
            send(f"Found ___`{len(picked_stocks)}`___ picked stocks\n\n")
            _pretty_print_stocks(picked_stocks)
        else:
            send("You don't have picked stocks currently")


def _pretty_print_stocks(stocks):
    table = Texttable()
    table.set_max_width(800)
    table.set_cols_align(["l", "l", "r", "r", "r", "r", "r", "r", "r"])
    table.set_cols_valign(["t", "t", "m", "m", "m", "m", "m", "m", "m"])
    headers = [
        "id",
        "name",
        "volume",
        "close",
        "diff",
        "change",
        "concen",
        "foreign",
        "trust",
    ]
    date = stocks[0][0]
    new_arr = [inner_arr[1:] for inner_arr in stocks]
    table.add_rows([headers] + new_arr)
    send(f"update from `{date}`\n```{table.draw()}```")


def _handle_file_operation_command(message):
    if "save:" in message:
        # parse the message to get the file name
        lines = message.split("save:")[1].split("\n")
        file_name = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        content = content.replace("```\n", "").replace("\n```", "")
        full_path = f"{STATIC_FOLDER}/tmp/{file_name}"
        with open(full_path, "w") as f:
            f.write(content)
        send(f"File [{file_name}](static/tmp/{file_name}) saved successfully.")
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


def _handle_system_command(message):
    if "reset_session" in message:
        if session.get("history", []):
            session["history"].clear()
        send("Session being reset successfully.")
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


def _truncate_text(text):
    max_length = 8192 - 2048
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length]
