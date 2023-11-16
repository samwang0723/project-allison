import os
import openai
import tiktoken
import time
import json
import inspect

from project_allison.constants import ENV_PATH
from project_allison.vectordb import query_vector_similarity

from dotenv import load_dotenv
from flask_socketio import send
from chromadb.api.models.Collection import Collection


COMPLETIONS_MODEL = "gpt-3.5-turbo"
ADVANCED_MODEL = "gpt-4-1106-preview"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_SECTION_LEN = 1024 * 3
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003
MIN_SIMILARITY = 0.75
MIN_DISTANCE = 0.4
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
    "fetch-jira-updates",
    # "image-to-text",
    # "query-knowledgebase",
    # "console-exeution",
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
            frame = inspect.currentframe()
            assert frame is not None
            print(f"{frame.f_code.co_name}, Error: {e}, retrying in 5 seconds")
            time.sleep(5)

    return ""


def construct_prompt(question: str, collection: Collection):
    chosen_sections = []
    chosen_sections_links = []
    deduped_attachments = []
    chosen_sections_attachments = []
    chosen_sections_len = 0
    similarities = []

    try:
        most_relevant_document_sections = _order_by_similarity(question, collection)
        documents = most_relevant_document_sections["documents"][0]
        metadatas = most_relevant_document_sections["metadatas"][0]
        distances = most_relevant_document_sections["distances"][0]

        for i, document_section in enumerate(documents):
            similarity = float(distances[i])
            title = metadatas[i]["title"]
            similarities.append(f"{title} - *{similarity}*")

            if similarity >= MIN_DISTANCE:
                continue

            tokens = int(metadatas[i]["num_tokens"])
            chosen_sections_len += tokens + SEPARATOR_LEN
            if chosen_sections_len > MAX_SECTION_LEN:
                print(
                    f"Max section length reached: {chosen_sections_len}, title: {title}, token: {tokens}"
                )
                break

            chosen_sections.append(str(SEPARATOR + document_section.replace("\n", " ")))

            chosen_sections_links.append(metadatas[i]["link"])
            if "attachments" in metadatas[i] and len(metadatas[i]["attachments"]) > 0:
                attachments = metadatas[i]["attachments"].split("|")
                trimmed_array = [s.strip() for s in attachments]
                if len(trimmed_array) > 0:
                    chosen_sections_attachments.extend(trimmed_array)
                    deduped_attachments = list(set(chosen_sections_attachments))
    except Exception as e:
        frame = inspect.currentframe()
        assert frame is not None
        print(f"{frame.f_code.co_name}, Error: {e}")

    if len(chosen_sections) == 0:
        prompt = ""
    else:
        prompt = _replace_slot(
            CONVERSATION_PROMPT,
            {"context": "".join(chosen_sections)},
        )

    return (prompt, chosen_sections_links, similarities, deduped_attachments)


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
    print(json_data)

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


def _order_by_similarity(query: str, collection: Collection):
    return query_vector_similarity(collection, query)


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
