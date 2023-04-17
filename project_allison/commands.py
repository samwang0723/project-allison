import uuid

from project_allison.constants import STATIC_FOLDER
from project_allison.downloader import download_content, get_plugin
from project_allison.chat_completion import (
    openai_call,
    ADVANCED_MODEL,
)

from flask import session
from flask_socketio import send
from texttable import Texttable

_system_commands = ["similarity", "prompt", "reset_session"]

TEXT_SUMMARY_PROMPT = (
    "Condense the text, KEEP critical details like time,person,action,amount."
)
EMAIL_SUMMARY_PROMPT = "[DO NOT CREATE RESPONSE] Condense email context with subject and summary, KEEP critical details like time,person,action,amount."
NEWS_SUMMARY_PROMPT = "Condense the news context with subject and summary, KEEP critical details like time,person,action,amount."


def handle_tasks(json_data):
    for item in json_data:
        task_name = item["task"]
        if task_name == "pull-my-stock-portfolio":
            _fetch_my_stock_portfolio()
            break
        elif task_name == "pull-stock-selections":
            send("Fetching stock selections is constructing.")
            break
        elif task_name == "fetch-jira-updates":
            arg = item["args"]
            if _validate(arg, task_name, ["text"]):
                _fetch_jira_updates(arg["text"])
            break
        elif task_name == "fetch-gmail-updates":
            _fetch_gmail_updates()
            break
        elif task_name == "fetch-news":
            _fetch_news()
            break
        elif task_name == "text-summary":
            arg = item["args"]
            if _validate(arg, task_name, ["text"]):
                _text_summary(arg["text"])
        elif task_name == "text-to-file":
            arg = item["args"]
            if _validate(arg, task_name, ["text"]):
                content = arg["text"]
                if "file" in arg:
                    file_name = arg["file"]
                else:
                    unique_id = uuid.uuid4()
                    file_name = str(unique_id) + ".txt"
                _text_to_file(file_name, content)
        else:
            send(f"Task `{task_name}` not found. Please try again.")


def handle_system_command(action):
    if action in _system_commands:
        if "reset_session" in action:
            if session.get("history", []):
                session["history"].clear()

            send(" Session being reset successfully.")
        else:
            _toggle_debugging(action)


def _validate(args, task_name, expected_keys):
    for key in expected_keys:
        if key not in args:
            send(f"No {key} found while performing {task_name}.")
            return False

    return True


def _toggle_debugging(action):
    if "similarity" in action:
        if session.get("similarity", False) == False:
            session["similarity"] = True
            send("Similarity scores will be shown.")
        else:
            session["similarity"] = False
            send("Similarity scores will be hidden.")
    elif "prompt" in action:
        if session.get("prompt", False) == False:
            session["prompt"] = True
            send("Prompt will be shown.")
        else:
            session["prompt"] = False
            send("Prompt will be hidden.")
    else:
        send("Command not found. Please try again.")


def _fetch_jira_updates(ticket_no):
    jira = get_plugin("JIRA")
    issues = jira.download(id=ticket_no)
    if len(issues) == 0:
        send("No ticket found.")
        return

    issue = issues[0]
    summary = issue.fields.summary
    status = issue.fields.status.name
    send(f"Title: `{summary}`, Status: `{status}`\n\n")

    child_issues = issues[1:]
    for child in child_issues:
        ticket_no = child.key
        summary = child.fields.summary
        status = child.fields.status.name
        link = jira.construct_link(id=ticket_no)
        send(f" - `{status}` ({ticket_no})[{link}] `{summary}`\n")

    comments = ""
    for c in issue.fields.comment.comments:
        comments += f"{c.author.displayName}:\n\t{c.body}\n"
    if comments != "":
        send(f"\n\nComments:\n\t - {comments}")


def _fetch_my_stock_portfolio():
    picked_stocks = download_content("finance::picked")
    if len(picked_stocks) > 0:
        send(f"Found ___`{len(picked_stocks)}`___ picked stocks\n\n")
        _pretty_print_stocks(picked_stocks)
    else:
        send("You don't have picked stocks currently")


def _fetch_gmail_updates():
    gmail_unread = download_content("gmail")
    if len(gmail_unread) > 0:
        send(f"You have ___`{len(gmail_unread)}`___ unread emails")
        for m in gmail_unread:
            send("\n\n---\n\n")
            _text_summary(_truncate_text(m["body"]), prompt=EMAIL_SUMMARY_PROMPT)
            send("\n\nSources:\n\t" + m["link"] + "\n")
    else:
        send("No unread emails.")


def _fetch_news():
    news = download_content("news")
    if len(news) > 0:
        send(f"Found ___`{len(news)}`___ news updates")
        for m in news:
            send("\n\n---\n\n")
            _text_summary(m["body"], prompt=NEWS_SUMMARY_PROMPT)
            send("\n\nSources:\n\t" + m["link"] + "\n")
    else:
        send("No unread news.")


def _text_summary(text, **kwargs) -> str:
    prompt = kwargs.get("prompt", None)
    if prompt is None:
        prompt = TEXT_SUMMARY_PROMPT

    res = openai_call(
        text,
        prompt,
        model=ADVANCED_MODEL,
        max_tokens=2048,
    )

    return res


def _text_to_file(file_name, content):
    full_path = f"{STATIC_FOLDER}/tmp/{file_name}"
    with open(full_path, "w") as f:
        f.write(content)

    send(f"File [{file_name}](static/tmp/{file_name}) saved successfully.")


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


def _truncate_text(text):
    max_length = 8192 - 2048
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length]
