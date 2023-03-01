# openai-training
Training example with openai

## Initialize python virtual environment

    $ python3 -m venv venv
    $ pip install -r requirements.txt

## Activate / Deactivate virtual env

    $ source ./venv/bin/activate
    $ deactivate

## Setup .env variables

    $ touch .env

### Paste all these three vars into .env

    CONFLUENCE_API_TOKEN={YOUR_CONFLUENCE_API_TOKEN}
    CONFLUENCE_API_USER={YOUR_CONFLUENCE_USER_NAME}
    OPENAI_API_KEY={YOUR_OPENAI_API_KEY}
