<div style="text-align:center;justify-content:center; width:100%; align-items:center; display:flex">
    <img src="https://user-images.githubusercontent.com/538559/228126732-c783e457-d6ba-47ea-8481-a05272c61ea8.png" alt="Project Allison" style="margin: 0 auto;" width="500"/>
</div>

# Project Allison
Project Allison is an assistant with openai, use document embeddings to understand domain knowledge, with streaming response

![Screenshot 2023-03-27 at 5 14 16 PM](https://user-images.githubusercontent.com/538559/227897967-03e771cf-9765-46df-986f-f634231ef9d3.png)

## Commands

    1. command:similarity
    2. command:prompt
    3. command:reset_session

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
    NEWS_API_KEY={YOUR_API_KEY}
    SKIP_GMAIL_SENDER={EMAILS}

## Put meaningful Confluence pages IDs into the pulling list
Make sure you have `source.csv` inside `project_allison/data` folder and listed category, type, page_id

## Setup Google Drive API
https://developers.google.com/drive/api/quickstart/python

Download `credentials.json` into `project_allison/auth` folder

Put source into CSV

    GOOGLE,{category},{page-id}

## Binary Builder
### build (make sure all the data in right place, make install will copy to ~/.project-allison)

    make install

### run
Please make sure you have all source files and credentials put under `$(HOME)/.project_allison`

    $ source ./venv/bin/activate
    $ project-allison --config-path=$HOME/.project_allison

### While failing doing make install with clang errors
$ brew install llvm@14

### Add customized clang into ~/.zshrc
export PATH="/opt/homebrew/Cellar/llvm@14/14.0.6/bin:$PATH"
