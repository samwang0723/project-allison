from project_allison.constants import STATIC_FOLDER
from project_allison.downloader import download_content
from project_allison.chat_completion import (
    openai_call,
    COMPLETIONS_MODEL,
    ADVANCED_MODEL,
)

from flask import session
from flask_socketio import send
from texttable import Texttable

_system_commands = ["similarity", "prompt", "reset_session"]


def handle_tasks(json_data):
    for item in json_data:
        print(
            f"Task: {item['task']}, ID: {item['id']}, Dep: {item['dep']}, Args: {item['args']}"
        )
        task_name = item["task"]
        if task_name == "pull-my-stock-portfolio":
            picked_stocks = download_content("finance::picked")
            if len(picked_stocks) > 0:
                send(f"Found ___`{len(picked_stocks)}`___ picked stocks\n\n")
                _pretty_print_stocks(picked_stocks)
            else:
                send("You don't have picked stocks currently")
            break
        elif task_name == "pull-stock-selections":
            send("Fetching stock selections is constructing.")
            break
        elif task_name == "fetch-gmail-updates":
            gmail_unread = download_content("gmail")
            if len(gmail_unread) > 0:
                send(f"You have ___`{len(gmail_unread)}`___ unread emails")
                for m in gmail_unread:
                    send("\n\n---\n\n")
                    openai_call(
                        _truncate_text(m["body"]),
                        "[DO NOT CREATE RESPONSE] Condense email context with subject and summary, KEEP critical details like time,person,action,amount.",
                        model=ADVANCED_MODEL,
                        max_tokens=2048,
                    )
                    send("\n\nSources:\n\t" + m["link"] + "\n")
            else:
                send("No unread emails.")
            break
        elif task_name == "fetch-news":
            news = download_content("news")
            if len(news) > 0:
                send(f"Found ___`{len(news)}`___ news updates")
                for m in news:
                    send("\n\n---\n\n")
                    openai_call(
                        _truncate_text(m["body"]),
                        "Condense the news context with subject and summary, KEEP critical details like time,person,action,amount.",
                        model=COMPLETIONS_MODEL,
                        max_tokens=1024,
                    )
                    send("\n\nSources:\n\t" + m["link"] + "\n")
            else:
                send("No unread news.")
            break
        elif task_name == "text-summary":
            openai_call(
                item["args"]["text"],
                "Condense the text, KEEP critical details like time,person,action,amount.",
                model=ADVANCED_MODEL,
                max_tokens=2048,
            )
        elif task_name == "text-to-file":
            arg = item["args"]
            file_name = arg["file"]
            content = arg["text"]
            full_path = f"{STATIC_FOLDER}/tmp/{file_name}"
            with open(full_path, "w") as f:
                f.write(content)
            send(f"File [{file_name}](static/tmp/{file_name}) saved successfully.")
        else:
            send(f"Task `{task_name}` not found. Please try again.")


def handle_system_command(action):
    if action in _system_commands:
        _handle_system_command(action)
    else:
        send("Command not found. Please try again.")


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
    previews = ""
    for stock in stocks:
        id = stock[1]
        previews += f"* https://stock.wearn.com/finance_chart.asp?stockid={id}&timekind=0&timeblock=120&sma1=8&sma2=21&sma3=55&volume=1\n"
    send(f"update from `{date}`\n```{table.draw()}```\n\n{previews}")


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
