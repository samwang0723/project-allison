from __future__ import print_function

import os.path
import base64

from jarvis.repository.plugin_interface import PluginInterface
from jarvis.constants import STORED_TOKEN, CREDENTIAL_TOKEN

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class Drive(PluginInterface):
    # If modifying these scopes, delete the file token.json.
    SCOPES = [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/gmail.readonly",
    ]

    def __init__(self):
        super().__init__()
        self.__creds = None
        self.__service = None
        self.__skip_gmails = os.getenv("SKIP_GMAIL_SENDER").split(",")

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

    def download(self, **kwargs) -> list:
        output = []
        if "file_type" in kwargs:
            file_type = kwargs["file_type"]
            if file_type == "gmail":
                output = self._fetch_gmail()
            else:
                if "file_id" in kwargs:
                    file_id = kwargs["file_id"]
                    output = self._fetch_gdrive(file_id)

        return output

    def construct_link(self, **kwargs) -> str:
        if "id" in kwargs:
            id = kwargs["id"]
            return "https://docs.google.com/document/d/" + id

        return ""

    def fetch_attachments(self, data) -> list:
        pass

    def _fetch_gdrive(self, file_id) -> list:
        output = []
        try:
            self.__service = build("drive", "v3", credentials=self.__creds)

            # Get the file metadata using the files().get() method with the fields parameter
            file_metadata = (
                self.__service.files().get(fileId=file_id, fields="id, name").execute()
            )
            # Extract the title from the file metadata
            file_title = file_metadata["name"]

            # Call the Drive v3 API
            results = (
                self.__service.files()
                .export_media(fileId=file_id, mimeType="text/html")
                .execute()
            )

            output.append(f"[title]{file_title}[/] {results.decode()}")
        except Exception as e:
            print(f"An error occurred: {e}")

        return output

    def _fetch_gmail(self):
        # Create a Gmail API client
        service = build("gmail", "v1", credentials=self.__creds)

        # Get all unread important emails
        output = []
        try:
            messages = (
                service.users()
                .messages()
                .list(userId="me", q="is:important is:unread", maxResults=10)
                .execute()
            )
            if "messages" in messages:
                for message in messages["messages"]:
                    msg = (
                        service.users()
                        .messages()
                        .get(userId="me", id=message["id"])
                        .execute()
                    )
                    headers = msg["payload"]["headers"]
                    for header in headers:
                        if header["name"] == "Subject":
                            title = header["value"]
                        if header["name"] == "From":
                            sender = header["value"]
                    if not self._skip_email_senders(sender):
                        try:
                            data = msg["payload"]["parts"][0]["body"]["data"]
                            byte_code = base64.urlsafe_b64decode(data)
                            body = self._extract_content(byte_code.decode("utf-8"))
                            id = msg["id"]
                            email_link = f"https://mail.google.com/mail/u/0/#inbox/{id}"

                            output.append(
                                {
                                    "link": email_link,
                                    "body": f"Subject: {title} ({sender}) \n {body}",
                                }
                            )
                        except Exception as e:
                            continue
        except Exception as e:
            print(f"An error occurred: {e}")

        return output

    def _extract_content(self, email_string):
        lines = email_string.split(">>")
        if len(lines) > 1:
            output = lines[0]
        else:
            output = email_string

        return output

    def _skip_email_senders(self, sender):
        for email in self.__skip_gmails:
            if email in sender:
                return True
        return False
