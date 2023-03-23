"""The main entry point. Invoke as `jarvis' or `python -m jarvis'.
"""
import pandas as pd
import numpy as np
import ast
import sys
import time
import subprocess
import pyperclip

from jarvis.voice_input import voice_recognition
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
from jarvis.constants import MATERIAL_FILE, VOICE_EXE
from jarvis.dynamic_console import console as _console

from collections import deque
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.console import group
from rich import box

USE_GPT_4 = "(gpt-4)"

_last_response = deque(maxlen=3)
_question_history = deque(maxlen=20)
_read_process = None


def _query(
    query: str,
    df: pd.DataFrame,
    show_prompt: bool = False,
    show_similarity: bool = False,
):
    start_time = time.time()
    prompt, links, similarities, attachments = construct_prompt(query, df)
    end_time = time.time()
    duration = end_time - start_time
    _console.print(
        "cosine_similarity - Duration:",
        duration,
        "seconds",
        style="bold red",
    )

    if show_similarity:
        for log in similarities:
            _console.print(log, style="bold green")

    prompt = "\n".join(_last_response) + prompt
    deduped_links = list(set(links))

    if show_prompt:
        _console.print("Prompt:\n\t" + prompt, style="bold green")

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
            start_time = time.time()
            response = chat_completion(
                prompt, query, model=model, max_tokens=max_tokens
            )
            end_time = time.time()
            duration = end_time - start_time
            _console.print(
                "openai.ChatCompletion - Duration:",
                duration,
                "seconds",
                style="bold red",
            )
            output = response["choices"][0]["message"]["content"].strip(" \n")

            return output
        except Exception as e:
            retries += 1
            _console.print(
                f"[[ Openai connection reset, wait for 5 secs ]]: {e}", style="bold red"
            )
            # If the connection is reset, wait for 5 seconds and retry
            time.sleep(5)

    return ""


def _extract_code(response) -> list:
    code = []
    parts = response.split("```")
    for i, part in enumerate(parts):
        if i % 2 == 1:
            code_lines = part.strip().split("\n", 1)
            code_block = code_lines[1] if len(code_lines) > 1 else code_lines[0]
            code.append(code_block)
    return code


def _should_read(command) -> bool:
    should_read = False
    if "Read it out" in command:
        should_read = True
    elif "Don't read" in command:
        should_read = False
    else:
        should_read = False

    cleaned_string = command.replace("Read it out", "").replace("Don't read", "")
    return cleaned_string.strip(), should_read


def _read(messages):
    _read_process = subprocess.Popen(
        ["python3", VOICE_EXE, messages], stdout=subprocess.PIPE
    )


def _reload_csv() -> pd.DataFrame:
    df = pd.read_csv(MATERIAL_FILE)
    df["embeddings"] = df["embeddings"].apply(lambda x: np.array(ast.literal_eval(x)))
    # Safely convert the 'attachments' column from string to list
    df["attachments"] = df["attachments"].apply(lambda x: ast.literal_eval(x))

    return df


def _helper_table() -> Table:
    table = Table(title="")
    table.add_column("Command", justify="middle", no_wrap=True)
    table.add_column("Description", justify="middle", no_wrap=True)
    table.add_row("[cyan bold]exit[/]", "exit the program")
    table.add_row("[cyan bold]show-prompt[/]", "show prompt for next conversation")
    table.add_row("[cyan bold]hide-prompt[/]", "hide prompt for next conversation")
    table.add_row("[cyan bold]clear[/]", "clear conversation history")
    table.add_row("[cyan bold]history[/]", "show conversation history")
    table.add_row("[cyan bold]help[/]", "show this help message")
    table.add_row("[cyan bold]show-similarity[/]", "show similarity")
    table.add_row("[cyan bold]hide-similarity[/]", "hide similarity")
    table.add_row("[cyan bold]voice (vv)[/]", "use voice input")
    table.add_row("[cyan bold]copy (cc)[/]", "copy code to clipboard")
    table.add_row("[cyan bold]gmail[/]", "get unread gmail")

    return table


@group()
def _print_result(response, links, attachments, read):
    yield Panel("Answer: ", style="bold green", box=box.SIMPLE)

    messages = ""
    parts = response.split("```")
    for i, part in enumerate(parts):
        if i % 2 == 1:  # Syntax-highlighted part
            code_lines = part.strip().split("\n", 1)
            code_type = code_lines[0] if len(code_lines) > 1 else "ruby"
            code_block = code_lines[1] if len(code_lines) > 1 else code_lines[0]
            yield Panel(
                Syntax(code_block, code_type, theme="monokai", line_numbers=True),
                box=box.SIMPLE,
            )
        else:  # Normal part
            message = part.strip("\n")
            messages = " ".join(message)

            yield Panel(message, box=box.SIMPLE)

    if len(links) > 0:
        table = Table(title="", box=box.SIMPLE)
        table.add_column("References", justify="middle", style="cyan", no_wrap=True)
        for l in links:
            table.add_row(l)

        yield Panel(table, box=box.SIMPLE)

    if len(attachments) > 0:
        table = Table(title="", box=box.SIMPLE)
        table.add_column(
            "Pdf Links for Reference", justify="middle", style="cyan", no_wrap=True
        )
        for a in attachments:
            table.add_row(a)

        yield Panel(table, box=box.SIMPLE)

    # calling voice to speaking
    # Execute the voice.py script with a command-line argument using Popen
    if read:
        _read(messages)


def main():
    try:
        pages = download_content(with_gdrive=True, with_confluence=True)
        df = get_dataframe(pages)
        df_with_embedding = inject_embeddings(df)
        df_with_embedding.to_csv(MATERIAL_FILE, index=False)

        # Reload CSV once to prevent formatting misalignment
        final_df = _reload_csv()

        # Listening input from user
        _prompt_on = False
        _print_similarity = False
        _read = False
        _extracted_code = []
        while True:
            # Note: Python 2.x users should use raw_input, the equivalent of 3.x's input
            _console.print("\n")
            command = _console.input("[cyan bold] Question / Command: [/]")
            question, _read = _should_read(command)
            if question == "exit":
                exit_status = ExitStatus.ERROR_CTRL_C
                break
            elif question == "show-prompt":
                _prompt_on = True
                _console.print(
                    "Prompt is now [red bold]on[/] for next conversation", style="cyan"
                )
                continue
            elif question == "hide-prompt":
                _prompt_on = False
                _console.print(
                    "Prompt is now [red bold]off[/] for next conversation", style="cyan"
                )
                continue
            elif question == "clear":
                _last_response.clear()
                _question_history.clear()
                _console.print("Conversation history cleared", style="cyan")
                continue
            elif question == "show-similarity":
                _print_similarity = True
                _console.print("Enable similarity", style="cyan")
                continue
            elif question == "hide-similarity":
                _print_similarity = False
                _console.print("Enable similarity", style="cyan")
                continue
            elif question == "history":
                table = Table(title="")
                table.add_column(
                    "History Records", justify="middle", style="cyan", no_wrap=True
                )
                for r in _question_history:
                    table.add_row(r)
                _console.print(table)
                continue
            elif question == "voice" or question == "vv":
                with _console.status("[bold green] Listening the voice"):
                    command = voice_recognition()
                    question, _read = _should_read(command)
                _console.print(f"[yellow bold] Command Received: [/] {command}")
            elif question == "copy" or question == "cc":
                if len(_extracted_code) > 0:
                    pyperclip.copy("\n\n".join(_extracted_code))
                    _console.print(f"[yellow bold] Code Copied to Clipboard [/]")
                else:
                    _console.print(
                        f"[yellow bold] No code found in the last response [/]"
                    )
                continue
            elif question == "gmail":
                gmail_unread = download_gmail()
                if len(gmail_unread) > 0:
                    _console.print(
                        f"[yellow bold] You have {len(gmail_unread)} Unread email threads [/]"
                    )
                    for m in gmail_unread:
                        output = _openai_call(
                            m["body"],
                            "Help to condense the email context with subject and summary, please not losing critical details",
                            model=ADVANCED_MODEL,
                            max_tokens=2048,
                        )
                        _console.print(Panel(_print_result(output, [m["link"]], _read)))
                else:
                    _console.print(f"[yellow bold] No unread emails [/]")
                continue
            elif question == "help":
                _console.print(_helper_table())
                continue

            response, links, attachments = _query(
                question, final_df, _prompt_on, _print_similarity
            )
            _extracted_code = _extract_code(response)
            _console.print(Panel(_print_result(response, links, attachments, _read)))
    except KeyboardInterrupt:
        exit_status = ExitStatus.ERROR_CTRL_C
        if _read_process is not None:
            _read_process.kill()

    return exit_status.value


if __name__ == "__main__":  # pragma: nocover
    sys.exit(main())
