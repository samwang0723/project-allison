#!/usr/bin/env python3

import os
import openai
import tiktoken
import random
import time
import pandas as pd
import numpy as np
import ast
import pyaudio
import wave
import re
import subprocess
import pytz
from datetime import datetime
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
face_proc = None


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

        header = """\n--\nPlease perform as a professional Crypto.com domain expert
        that can answer questions about Crypto.com specific knowledge giving below
        context"""
        prompt = "Context: " + "".join(chosen_sections)
        prompt = "\n".join(self.last_response) + prompt + header

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


def face_recognition():
    global face_proc
    face_proc = subprocess.Popen(["python3", "face_detect.py"])

    while True:
        try:
            # Check if the voice output process has completed
            return_code = face_proc.poll()
            if return_code is not None:
                break
        except:
            console.print("Face recognition process is not running", style="bold red")


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
        with console.status("[bold green] Listening the voice"):
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
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

        # use ffmpeg to suppress noise and increase volume of voice
        not_having_noise = noise_suppression()
        if not_having_noise == False:
            return ""

        audio_file = open("./voice_records/final.wav", "rb")
        # transcribe turns into all languages, instead translate only to english
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

    except Exception as e:
        console.print(e, style="bold red")
        transcript = {"text": ""}

    return transcript["text"]


def noise_suppression():
    p = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "quiet",
            "-i",
            WAVE_OUTPUT_FILENAME,
            "-af",
            "afftdn=nf=-25",
            "./voice_records/output.wav",
        ]
    )
    p.communicate()

    p = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "quiet",
            "-i",
            "./voice_records/output.wav",
            "-af",
            "afftdn=nf=-25",
            "./voice_records/output2.wav",
        ]
    )
    p.communicate()

    p = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "quiet",
            "-i",
            "./voice_records/output2.wav",
            "-af",
            "highpass=f=200, lowpass=f=3000",
            "./voice_records/output3.wav",
        ]
    )
    p.communicate()

    p = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "quiet",
            "-i",
            "./voice_records/output3.wav",
            "-af",
            "volume=4",
            "./voice_records/final.wav",
        ]
    )
    p.communicate()

    command = "ffmpeg -hide_banner -stats -i ./voice_records/final.wav -af silencedetect=noise=0dB:d=3 -vn -sn -dn -f null -"
    out = subprocess.Popen(
        command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    stdout, stderr = out.communicate()
    if stderr is not None:
        return False
    output = stdout.decode("utf-8")

    # detect silent duration
    cmd = "ffmpeg -hide_banner -stats -i ./voice_records/final.wav -af silencedetect=noise=-50dB:d=3 -vn -sn -dn -f null -"
    params = cmd.split()
    voice_proc = subprocess.Popen(
        params, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    stdout, _ = voice_proc.communicate()
    output = stdout.decode("utf-8")

    pattern = r"silence_duration: (\d+\.\d+)"
    matches = re.findall(pattern, output)
    for match in matches:
        console.print(f"Silence duration: {match}", style="bold red")
        if float(match) > 7:
            return False

    return True


def greeting() -> str:
    native_datetime = datetime.now()
    timezone = pytz.timezone("Asia/Taipei")
    lt = timezone.localize(native_datetime)
    current_time = lt.time()

    time_str = None
    if current_time.hour < 12:
        time_str = "Good morning"
    elif current_time.hour < 17:
        time_str = "Good afternoon"
    elif current_time.hour < 20:
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
    # Format the time object as a string
    formatted_time = current_time.strftime("%I:%M%p")

    return f"{time_str} Sir, it's {formatted_time}, {random_greeting}"


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
            and time.time() - question_starting_time > 60 * 5
        ):
            console.print(
                "Voice input is now [red bold]off[/] until you call Jarvis again",
                style="cyan",
            )
            question_starting_time = None

        if question_starting_time is None:
            face_recognition()
            proc = subprocess.Popen(["python3", "voice.py", greeting()])
            question_starting_time = time.time()
            continue
        else:
            question = voice_recognition()
            if len(question) < 5:
                console.print("[red bold]Cannot detect voice input[/]")
                time.sleep(3)
                continue
            else:
                console.print(f"Voice Transcript: {question}")

        # question = console.input("[cyan bold] Question / Command: [/]")
        if "Terminate the program" in question:
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
