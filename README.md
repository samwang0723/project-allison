# openai-training
Training example with openai

## Initialize python virtual environment

    $ python3 -m venv venv
    $ source ./venv/bin/activate

### Install dependency libraries (using Mac M1/M2)
In this case we want to support audio input/output, need to do some of the customization

    $ brew install portaudio
    $ brew install ffmpeg
    $ brew install flac
    (vnev)$ pip install pyaudio
    (venv)$ pip install --upgrade wheel
    (venv)$ pip install -r requirements.txt

While you tried to run the audio program on Mac, you probably will meet

    /objc/_convenience.py", line 134, in container_unwrap raise exc_type(*exc_args) KeyError: 'VoiceAge'

Please follow the instruction here to solve it https://github.com/nateshmbhat/pyttsx3/issues/248#issuecomment-1342334549

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

### Put meaningful Confluence pages IDs into the pulling list
Make sure you have `source.csv` inside `data` folder and listed category, type, page_id

### Setup Google Drive API
https://developers.google.com/drive/api/quickstart/python

Download `credentials.json` into `/auth` folder

Put source into CSV

    GOOGLE,{category},{page-id}

## Execution

    (venv)$ python3 knowledge_base.py

## Docker

### build

    docker build -t knowledge_base .

### run

    docker run --env-file .env -v $(pwd)/data:/app/data -it knowledge_base
