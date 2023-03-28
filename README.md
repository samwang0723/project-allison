# Project Allison
Project Allison is an assistant with openai, use document embeddings to understand domain knowledge, with streaming response

<img src="https://user-images.githubusercontent.com/538559/228126732-c783e457-d6ba-47ea-8481-a05272c61ea8.png" alt="Project Allison" width="500"/>

![Screenshot 2023-03-27 at 5 14 16 PM](https://user-images.githubusercontent.com/538559/227897967-03e771cf-9765-46df-986f-f634231ef9d3.png)

## Commands

    1. command:fetch_gmail
    2. command:show_similarity
    3. command:hidden_similarity
    4. command:show_prompt
    5. command:hide_prompt
    6. command:reload_csv
    7. command:reset_session

### File operations

command:save:{file_name}

    command:save:{file_name}
    ```
    {text, code, etc...}
    ```

command:diagram:

    command:diagram:
    ```
    {dot format code}
    ```


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
