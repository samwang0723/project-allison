"""The main entry point. Invoke as `jarvis' or `python -m jarvis'.
"""
import pandas as pd
import numpy as np
import ast
import sys
import time
import subprocess

from jarvis.voice_input import voice_recognition
from jarvis.tokenizer import get_dataframe
from jarvis.downloader import download_content
from jarvis.chat import chat_completion, inject_embeddings
from jarvis.status import ExitStatus
from jarvis.chat import construct_prompt
from jarvis.constants import MATERIAL_FILE, VOICE_EXE
from jarvis.dynamic_console import console as _console

from collections import deque
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.console import group
from rich import box


_last_response = deque(maxlen=3)
_question_history = deque(maxlen=20)


def _reload_csv():
    df = pd.read_csv(MATERIAL_FILE)
    df["embeddings"] = df["embeddings"].apply(lambda x: np.array(ast.literal_eval(x)))

    return df


def _query(
    query: str,
    df: pd.DataFrame,
    show_prompt: bool = False,
    show_similarity: bool = False,
):
    start_time = time.time()
    prompt, links, similarities = construct_prompt(query, df)
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
    try:
        start_time = time.time()
        response = chat_completion(prompt, query)
        end_time = time.time()
        duration = end_time - start_time
        _console.print(
            "openai.ChatCompletion - Duration:",
            duration,
            "seconds",
            style="bold red",
        )
    except:
        _console.print(
            "[[ Openai connection reset, wait for 5 secs ]]", style="bold red"
        )
        # If the connection is reset, wait for 5 seconds and retry
        time.sleep(5)
        response = chat_completion(prompt, query)

    output = response["choices"][0]["message"]["content"].strip(" \n")

    _last_response.append(output)
    _question_history.append(query)

    return output, deduped_links


@group()
def _print_result(response, links, read):
    yield Panel("Answer: ", style="bold green", box=box.SIMPLE)

    messages = ""
    parts = response.split("```")
    for i, part in enumerate(parts):
        if i % 2 == 1:  # Syntax-highlighted part
            yield Panel(
                Syntax(part, "ruby", theme="monokai", line_numbers=True), box=box.SIMPLE
            )
        else:  # Normal part
            message = part.strip("\n")
            messages += " " + message

            yield Panel(message, box=box.SIMPLE)

    if len(links) > 0:
        table = Table(title="", box=box.SIMPLE)
        table.add_column("References", justify="middle", style="cyan", no_wrap=True)
        for l in links:
            table.add_row(l)

        yield Panel(table, box=box.SIMPLE)

    # calling voice to speaking
    # Execute the voice.py script with a command-line argument using Popen
    if read:
        subprocess.Popen(["python3", VOICE_EXE, messages], stdout=subprocess.PIPE)


def should_read(command):
    should_read = False
    if "Read it out" in command:
        should_read = True
    elif "Don't read" in command:
        should_read = False
    else:
        should_read = False

    cleaned_string = command.replace("Read it out", "").replace("Don't read", "")
    return cleaned_string.strip(), should_read


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
        while True:
            # Note: Python 2.x users should use raw_input, the equivalent of 3.x's input
            _console.print("\n")
            command = _console.input("[cyan bold] Question / Command: [/]")
            if command == "exit":
                exit_status = ExitStatus.ERROR_CTRL_C
                break
            elif command == "show-prompt":
                _prompt_on = True
                _console.print(
                    "Prompt is now [red bold]on[/] for next conversation", style="cyan"
                )
                continue
            elif command == "hide-prompt":
                _prompt_on = False
                _console.print(
                    "Prompt is now [red bold]off[/] for next conversation", style="cyan"
                )
                continue
            elif command == "clear":
                _last_response.clear()
                _question_history.clear()
                _console.print("Conversation history cleared", style="cyan")
                continue
            elif command == "show-similarity":
                _print_similarity = True
                _console.print("Enable similarity", style="cyan")
                continue
            elif command == "hide-similarity":
                _print_similarity = False
                _console.print("Enable similarity", style="cyan")
                continue
            elif command == "history":
                table = Table(title="")
                table.add_column(
                    "History Records", justify="middle", style="cyan", no_wrap=True
                )
                for r in _question_history:
                    table.add_row(r)
                _console.print(table)
                continue
            elif command == "voice" or command == "vv":
                with _console.status("[bold green] Listening the voice"):
                    command = voice_recognition()
                _console.print(f"[yellow bold] Command Received: [/] {command}")
            elif command == "help":
                table = Table(title="")
                table.add_column("Command", justify="middle", no_wrap=True)
                table.add_column("Description", justify="middle", no_wrap=True)
                table.add_row("[cyan bold]exit[/]", "exit the program")
                table.add_row(
                    "[cyan bold]show-prompt[/]", "show prompt for next conversation"
                )
                table.add_row(
                    "[cyan bold]hide-prompt[/]", "hide prompt for next conversation"
                )
                table.add_row("[cyan bold]clear[/]", "clear conversation history")
                table.add_row("[cyan bold]history[/]", "show conversation history")
                table.add_row("[cyan bold]help[/]", "show this help message")
                table.add_row("[cyan bold]show-similarity[/]", "show similarity")
                table.add_row("[cyan bold]hide-similarity[/]", "hide similarity")
                table.add_row("[cyan bold]voice (vv)[/]", "use voice input")

                _console.print(table)
                continue

            question, _read = should_read(command)
            response, links = _query(question, final_df, _prompt_on, _print_similarity)
            _console.print(Panel(_print_result(response, links, _read)))
    except KeyboardInterrupt:
        exit_status = ExitStatus.ERROR_CTRL_C

    return exit_status.value


if __name__ == "__main__":  # pragma: nocover
    sys.exit(main())
