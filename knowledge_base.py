#!/usr/bin/env python3

import os
import openai
import tiktoken
import datetime
import random
import time
import pandas as pd
import numpy as np
import ast
import pyaudio
import wave
import subprocess
from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv
from confluence import Wiki
from collections import deque
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.console import group
from rich import box

console = Console(width=120)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 8
WAVE_OUTPUT_FILENAME = "./voice_records/voice.wav"
proc = None


class KnowledgeBase:
    COMPLETIONS_MODEL = "gpt-3.5-turbo"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because
        # it gives the most predictable, factual answer.
        "temperature": 0.0,
        "max_tokens": 1024,
        "model": COMPLETIONS_MODEL,
        "top_p": 0.1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    MAX_SECTION_LEN = 2046
    SEPARATOR = "\n* "
    ENCODING = "gpt2"  # encoding for text-davinci-003
    MATERIAL_FILE = "./data/material.csv"
    MIN_SIMILARITY = 0.8

    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.encoding = tiktoken.get_encoding(self.ENCODING)
        self.separator_len = len(self.encoding.encode(self.SEPARATOR))
        self.last_response = deque(maxlen=3)
        self.question_history = deque(maxlen=20)
        self.print_similarity = False

    def answer_query_with_context(
        self,
        query: str,
        df: pd.DataFrame,
        show_prompt: bool = False,
    ):
        global proc
        prompt, links = self.__construct_prompt(query, df)
        deduped_links = list(set(links))

        if show_prompt:
            console.print("Prompt:\n\t" + prompt, style="bold green")
        try:
            start_time = time.time()
            response = self.__chat_completion(prompt, query)
            end_time = time.time()
            duration = end_time - start_time
            console.print(
                "openai.ChatCompletion - Duration:",
                duration,
                "seconds",
                style="bold red",
            )
        except:
            console.print(
                "[[ Openai connection reset, wait for 5 secs ]]", style="bold red"
            )
            proc = subprocess.Popen(
                [
                    "python3",
                    "voice.py",
                    "Seems like openai server is hitting rate limit, please wait a moment",
                ]
            )
            # If the connection is reset, wait for 5 seconds and retry
            time.sleep(5)
            response = self.__chat_completion(prompt, query)

        output = response["choices"][0]["message"]["content"].strip(" \n")
        self.last_response.append(output)
        self.question_history.append(query)

        return output, deduped_links

    def calc_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        if "embeddings" in df.columns:
            mask = df["embeddings"].isna()
            df.loc[mask, "embeddings"] = df.loc[mask, "body"].apply(
                lambda x: get_embedding(x, engine=self.EMBEDDING_MODEL)
            )
        else:
            df["embeddings"] = df["body"].apply(
                lambda x: get_embedding(x, engine=self.EMBEDDING_MODEL)
            )
        return df

    def __chat_completion(self, prompt: str, query: str):
        return openai.ChatCompletion.create(
            **self.COMPLETIONS_API_PARAMS,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )

    def __order_document_sections_by_query_similarity(
        self, query: str, df: pd.DataFrame
    ):
        start_time = time.time()
        query_embedding = get_embedding(query, engine=self.EMBEDDING_MODEL)
        df["similarity"] = df.embeddings.apply(
            lambda x: cosine_similarity(x, query_embedding)
        )

        results = df.sort_values("similarity", ascending=False).head(3)
        end_time = time.time()
        duration = end_time - start_time
        console.print(
            "cosine_similarity - Duration:",
            duration,
            "seconds",
            style="bold red",
        )

        return results

    def __construct_prompt(self, question: str, df: pd.DataFrame):
        most_relevant_document_sections = (
            self.__order_document_sections_by_query_similarity(question, df)
        )

        chosen_sections = []
        chosen_sections_links = []
        chosen_sections_len = 0

        for _, document_section in most_relevant_document_sections.iterrows():
            if self.print_similarity:
                console.print(
                    f"{document_section.title} - {document_section.similarity}",
                    style="bold green",
                )
            if document_section.similarity < self.MIN_SIMILARITY:
                continue

            chosen_sections_len += int(document_section.num_tokens) + self.separator_len
            if chosen_sections_len > self.MAX_SECTION_LEN:
                break

            chosen_sections.append(
                str(self.SEPARATOR + document_section.body.replace("\n", " "))
            )
            chosen_sections_links.append(document_section.link)

        header = """Please perform as a professional Crypto.com domain expert
        that can answer questions about Crypto.com specific knowledge giving below
        context.\n\nContext:\n"""
        prompt = header + "".join(chosen_sections)
        prompt = "\n".join(self.last_response) + prompt

        return (prompt, chosen_sections_links)


def parse_numbers(s):
    return [float(x) for x in s.strip("[]").split(",")]


def update_internal_doc_embeddings(kb: KnowledgeBase) -> pd.DataFrame:
    console.print(
        "Updating internal document embeddings from Confluence...", style="bold red"
    )

    wiki = Wiki()
    confluence = wiki.connect_to_confluence()
    # gDrive = wiki.connect_to_drive()
    pages = wiki.get_all_pages_from_ids(confluence)
    df = wiki.collect_with_processes(pages)
    df = kb.calc_embeddings(df)
    df.to_csv(kb.MATERIAL_FILE, index=False)

    # to avoid new/old embeddings format difference
    new_df = pd.read_csv(kb.MATERIAL_FILE)
    new_df["embeddings"] = new_df["embeddings"].apply(
        lambda x: np.array(ast.literal_eval(x))
    )

    console.print("Confluence download and index completed!", style="bold yellow")

    return new_df


@group()
def print_result(response, links):
    global proc
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
    proc = subprocess.Popen(["python3", "voice.py", messages])


def voice_recognition():
    global proc
    # If voice still speaking, don't listen and parse the transcript
    while True:
        try:
            # Check if the voice output process has completed
            return_code = proc.poll()
            if return_code is not None:
                break
        except:
            break

    try:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            frames_per_buffer=CHUNK,
            input=True,
        )

        frames = []
        with console.status("[bold green] Listening the voice") as status:
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                # Convert data to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                # Compute energy of audio frame
                energy = np.sum(audio_data**2) / len(audio_data)
                # If energy is above threshold, do something with the audio data
                if energy >= 1500:
                    # Process audio data here
                    frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()

        audio_file = open(WAVE_OUTPUT_FILENAME, "rb")
        # transcribe turns into all languages, instead translate only to english
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    except:
        transcript = {"text": ""}

    return transcript["text"]


def greeting() -> str:
    current_time = datetime.datetime.now().time()
    time_str = None
    if current_time < datetime.time(12):
        time_str = "Good morning"
    elif current_time < datetime.time(17):
        time_str = "Good afternoon"
    elif current_time < datetime.time(20):
        time_str = "Good evening"
    else:
        time_str = "Greeting"

    greetings = [
        "It's always a pleasure to see you.",
        "How may I assist you today",
        "How's your day going?",
        "I hope you're having a fantastic day so far.",
        "How are you doing today?",
    ]
    random_greeting = random.choice(greetings)

    return f"{time_str} Sir, {random_greeting}"


def main():
    load_dotenv()
    kb = KnowledgeBase()
    df = update_internal_doc_embeddings(kb)
    prompt_on = False
    global proc

    question_starting_time = None

    while True:
        if (
            question_starting_time is not None
            and time.time() - question_starting_time > 30
        ):
            console.print(
                "Voice input is now [red bold]off[/] until you call Jarvis again",
                style="cyan",
            )
            question_starting_time = None
        # Note: Python 2.x users should use raw_input, the equivalent of 3.x's input
        console.print("\n")
        question = voice_recognition()
        if len(question) < 5:
            continue
        else:
            console.print(f"Voice Transcript: {question}")

        # question = console.input("[cyan bold] Question / Command: [/]")
        if "Jarvis" in question:
            proc = subprocess.Popen(["python3", "voice.py", greeting()])
            question_starting_time = time.time()
            continue
        elif question == "Exit":
            break
        elif "Show prompt" in question:
            prompt_on = True
            console.print(
                "Prompt is now [red bold]on[/] for next conversation", style="cyan"
            )
            continue
        elif "Hide prompt" in question:
            prompt_on = False
            console.print(
                "Prompt is now [red bold]off[/] for next conversation", style="cyan"
            )
            continue
        elif "Clear history" in question:
            kb.last_response.clear()
            kb.question_history.clear()
            console.print("Conversation history cleared", style="cyan")
            continue
        elif "Show similarity" in question:
            kb.print_similarity = True
            console.print("Similarity is now [red bold]on[/]", style="cyan")
            continue
        elif "Hide similarity" in question:
            kb.print_similarity = False
            console.print("Similarity is now [red bold]off[/]", style="cyan")
            continue
        elif "List History" in question:
            table = Table(title="")
            table.add_column(
                "History Records", justify="middle", style="cyan", no_wrap=True
            )
            for r in kb.question_history:
                table.add_row(r)

            console.print(table)
            continue
        elif question == "help":
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

            console.print(table)
            continue
        elif "Jarvis" not in question and question_starting_time is not None:
            question_starting_time = time.time()
            response, links = kb.answer_query_with_context(question, df, prompt_on)
            console.print(Panel(print_result(response, links)))

            prompt_on = False


if __name__ == "__main__":
    main()
