# openai-training
Training example with openai

## Initialize python virtual environment

    $ python3 -m venv venv
    $ source ./venv/bin/activate
    (venv)$ pip install -r requirements.txt

### Refresh all installed library into requirements.txt

    (venv)$ pip freeze > requirements.txt

## Activate / Deactivate virtual env

    $ source ./venv/bin/activate
    (venv)$ deactivate

## Setup .env variables

    $ touch .env

### Paste all these three vars into .env

    CONFLUENCE_API_TOKEN={YOUR_CONFLUENCE_API_TOKEN}
    CONFLUENCE_API_USER={YOUR_CONFLUENCE_USER_NAME}
    OPENAI_API_KEY={YOUR_OPENAI_API_KEY}

## Execution

    (venv)$ python3 knowledge_base.py

### Put meaningful Confluence pages IDs into the pulling list
Make sure you have `source.csv` inside `data` folder and listed category, type, page_id
