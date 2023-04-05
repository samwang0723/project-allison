import os
import openai
import tiktoken
import time
import json
import pandas as pd

from project_allison.constants import ENV_PATH

from dotenv import load_dotenv
from openai.embeddings_utils import get_embedding, cosine_similarity
from flask_socketio import send

COMPLETIONS_MODEL = "gpt-3.5-turbo"
ADVANCED_MODEL = "gpt-4"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_SECTION_LEN = 2046
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003
MIN_SIMILARITY = 0.75
SEPARATOR_LEN = len(tiktoken.get_encoding(ENCODING).encode(SEPARATOR))
CONVERSATION_PROMPT = """
#1 The AI assistant can parse user input and answer questions based on context given. You need to make sure all the code MUST wrapped inside 
```(code-language)
(code)
```
if response has simplified chinese, you MUST convert to traditional chinese.
Context: [ {{context}} ]
"""
TASK_PREPARATION_PROMPT = """#1 Task Planning Stage: The AI assistant can parse user input to several tasks: [{"task": task, "id", task_id, "dep": dependency_task_id, "args": {"text": text or <GENERATED>-dep_id, "image": image_url or <GENERATED>-dep_id,"audio": audio_url or <GENERATED>-dep_id,"file": file_path or <GENERATED>-dep_id}}]. The special tag "<GENERATED>-dep_id" refer to the one generated text/image/audio/file in the dependency task (Please consider whether the dependency task generates resources of this type.) and "dep_id" must be in "dep" list. The "dep" field denotes the ids of the previous prerequisite tasks which generate a new resource that the current task relies on. The "args" field must in ["text", "image", "audio", "file"], nothing else. The task MUST selected from the following options: {{available_tasks}}. If no task options is suitable, make the task as "none" and stop. There may be multiple tasks of the same type. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. If the user input can't be parsed, you need reply empty JSON []. If "text-to-file" tasks found, recognize text inside ``` as text content and KEEP NEW LINE, NOT parsing it."""
TASK_CHAT_PROMPT = "The chat log [ {{context}} ] may contain the resources I mentioned. Now I input { {{input}} }, please parse out as many as the required tasks to solve my request ONLY in a JSON format without any description."
AVAILABLE_TASKS = [
    "pull-my-stock-portfolio",
    "pull-stock-selections",
    "fetch-gmail-updates",
    "fetch-news",
    "text-summary",
    "text-to-file",
    "text-to-diagram",
    "image-to-text",
    "query-knowledgebase",
    "console-exeution",
]


def openai_call(prompt, query, model=COMPLETIONS_MODEL, max_tokens=1024) -> str:
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            response = _chat_completion(
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


def construct_prompt(question: str, df: pd.DataFrame):
    most_relevant_document_sections = _order_by_similarity(question, df)

    chosen_sections = []
    chosen_sections_links = []
    deduped_attachments = []
    chosen_sections_attachments = []
    chosen_sections_len = 0
    similarities = []

    for _, document_section in most_relevant_document_sections.iterrows():
        similarity = document_section.similarity
        if similarity >= MIN_SIMILARITY:
            similarity = f"**{similarity}**"
        similarities.append(f"{document_section.title} - {similarity}")
        if document_section.similarity < MIN_SIMILARITY:
            continue

        chosen_sections_len += int(document_section.num_tokens) + SEPARATOR_LEN
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(
            str(SEPARATOR + document_section.body.replace("\n", " "))
        )

        chosen_sections_links.append(document_section.link)
        if len(document_section.attachments) > 0:
            chosen_sections_attachments.extend(document_section.attachments)
            deduped_attachments = list(set(chosen_sections_attachments))

    if len(chosen_sections) == 0:
        prompt = ""
    else:
        prompt = _replace_slot(
            CONVERSATION_PROMPT,
            {"context": "".join(chosen_sections)},
        )

    return (prompt, chosen_sections_links, similarities, deduped_attachments)


def inject_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    if "embeddings" in df.columns:
        mask = df["embeddings"].isna()
        df.loc[mask, "embeddings"] = df.loc[mask, "body"].apply(
            lambda x: get_embedding(x, engine=EMBEDDING_MODEL)
        )
    else:
        df["embeddings"] = df["body"].apply(
            lambda x: get_embedding(x, engine=EMBEDDING_MODEL)
        )
    return df


def parse_task_prompt(query, history):
    polished_input = _replace_slot(
        TASK_CHAT_PROMPT,
        {
            "input": query,
            "context": history,
        },
    )
    preparation_prompt = _replace_slot(
        TASK_PREPARATION_PROMPT,
        {
            "available_tasks": ",".join(AVAILABLE_TASKS),
        },
    )

    resp = openai.ChatCompletion.create(
        **_task_params(model=COMPLETIONS_MODEL, max_tokens=1024),
        messages=[
            {"role": "system", "content": preparation_prompt},
            {"role": "user", "content": polished_input},
        ],
        stream=False,
    )

    json_data = json.loads(resp["choices"][0]["message"]["content"])
    return json_data


def _chat_completion(
    prompt: str, query: str, model: str = COMPLETIONS_MODEL, max_tokens: int = 1024
):
    return openai.ChatCompletion.create(
        **_conversation_params(model=model, max_tokens=max_tokens),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        stream=True,
    )


def _order_by_similarity(query: str, df: pd.DataFrame):
    query_embedding = get_embedding(query, engine=EMBEDDING_MODEL)
    df["similarity"] = df.embeddings.apply(
        lambda x: cosine_similarity(x, query_embedding)
    )
    results = df.sort_values("similarity", ascending=False).head(3)
    return results


def _replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace(
            "{{" + key + "}}", value.replace('"', "'")  # .replace("\n", "")
        )
    return text


def _task_params(model: str = COMPLETIONS_MODEL, max_tokens: int = 1024):
    return {
        "temperature": 0,
        "max_tokens": max_tokens,
        "model": model,
    }


def _conversation_params(model: str = COMPLETIONS_MODEL, max_tokens: int = 1024):
    return {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "model": model,
        "top_p": 0.1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }


def _init():
    load_dotenv(dotenv_path=ENV_PATH)
    openai.api_key = os.environ["OPENAI_API_KEY"]


_init()
