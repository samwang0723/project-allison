# openai-training
Project Allison is an assistant with openai, use document embeddings to understand domain knowledge, with streaming response


## Setup .env variables

    $ touch .env

## Paste all these three vars into .env

    CONFLUENCE_API_TOKEN={YOUR_CONFLUENCE_API_TOKEN}
    CONFLUENCE_API_USER={YOUR_CONFLUENCE_USER_NAME}
    OPENAI_API_KEY={YOUR_OPENAI_API_KEY}
    CONFLUENCE_HOST={YOUR_HOST}
    SKIP_SSL_VERIFICATION=0
    TOKENIZERS_PARALLELISM=false
    FLASK_SECRET_KEY={YOUR_SECRET}

## Put meaningful Confluence pages IDs into the pulling list
Make sure you have `source.csv` inside `jarvis/data` folder and listed category, type, page_id

## Setup Google Drive API
https://developers.google.com/drive/api/quickstart/python

Download `credentials.json` into `jarvis/auth` folder

Put source into CSV

    GOOGLE,{category},{page-id}

## Binary Builder
### build (make sure all the data in right place, make install will copy to ~/.jarvis)

    make install

### run
Please make sure you have all source files and credentials put under `$(HOME)/.jarvis`

    $ source ./venv/bin/activate
    $ jarvis --config-path=$HOME/.jarvis
