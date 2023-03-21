from __future__ import print_function

import os.path

from jarvis.constants import STORED_TOKEN, CREDENTIAL_TOKEN

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class Drive:
    # If modifying these scopes, delete the file token.json.
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

    def __init__(self):
        self.__creds = None
        self.__service = None

    def authenticate(self):
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(STORED_TOKEN):
            self.__creds = Credentials.from_authorized_user_file(
                STORED_TOKEN, self.SCOPES
            )
        # If there are no (valid) credentials available, let the user log in.
        if not self.__creds or not self.__creds.valid:
            if self.__creds and self.__creds.expired and self.__creds.refresh_token:
                self.__creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIAL_TOKEN, self.SCOPES
                )
                self.__creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(STORED_TOKEN, "w") as token:
                token.write(self.__creds.to_json())

    def download_file(self, fileId, mimeType="text/html") -> str:
        output = ""
        try:
            self.__service = build("drive", "v3", credentials=self.__creds)

            # Get the file metadata using the files().get() method with the fields parameter
            file_metadata = (
                self.__service.files().get(fileId=fileId, fields="id, name").execute()
            )
            # Extract the title from the file metadata
            file_title = file_metadata["name"]

            # Call the Drive v3 API
            results = (
                self.__service.files()
                .export_media(fileId=fileId, mimeType=mimeType)
                .execute()
            )

            output = f"[title]{file_title}[/] {results.decode()}"
        except HttpError as error:
            # TODO(developer) - Handle errors from drive API.
            print(f"An error occurred: {error}")

        return output

    def get_link(self, id) -> str:
        return "https://docs.google.com/document/d/" + id
