import os
import requests
import json

from jarvis.constants import ENV_PATH

from dotenv import load_dotenv
from atlassian import Confluence


class Wiki:
    def __init__(self):
        load_dotenv(dotenv_path=ENV_PATH)
        self.host = os.environ["CONFLUENCE_HOST"]
        self.username = os.environ["CONFLUENCE_API_USER"]
        self.access_token = os.environ["CONFLUENCE_API_TOKEN"]
        self.api = None
        self.auth = (self.username, self.access_token)

    def authenticate(self):
        self.api = Confluence(
            url=self.host,
            username=self.username,
            password=self.access_token,
            cloud=True,
        )

    def get_attachments(self, page_id):
        supported_types = ["application/pdf"]
        attachments_url = (
            f"{self.host}/wiki/rest/api/content/{page_id}/child/attachment"
        )
        response = requests.get(attachments_url, auth=self.auth)

        if response.status_code == 200:
            attachments = json.loads(response.text)
            links = []

            # Iterate through the attachments to find PDF files
            for attachment in attachments["results"]:
                if attachment["extensions"]["mediaType"] in supported_types:
                    download_url = attachment["_links"]["download"]
                    links.append(f"{self.host}/wiki{download_url}")

            return links

    def get_link(self, id, space) -> str:
        return self.host + "/wiki/spaces/" + space + "/pages/" + id
