import os
import requests
import json

from project_allison.plugins.plugin_interface import PluginInterface

from atlassian import Confluence


class Wiki(PluginInterface):
    def __init__(self):
        super().__init__()

        self.__host = os.environ["CONFLUENCE_HOST"]
        self.__username = os.environ["CONFLUENCE_API_USER"]
        self.__access_token = os.environ["CONFLUENCE_API_TOKEN"]
        if os.environ["SKIP_SSL_VERIFICATION"] == "1":
            self.__verify_ssl = False
        else:
            self.__verify_ssl = True
        self.__api = None
        self.__auth = (self.__username, self.__access_token)

    def authenticate(self):
        self.__api = Confluence(
            url=self.__host,
            username=self.__username,
            password=self.__access_token,
            cloud=True,
            verify_ssl=self.__verify_ssl,
        )

    def fetch_attachments(self, data) -> list:
        supported_types = ["application/pdf", "image/png", "image/jpeg", "image/gif"]
        attachments_url = f"{self.__host}/wiki/rest/api/content/{data}/child/attachment"
        response = requests.get(
            attachments_url, auth=self.__auth, verify=self.__verify_ssl
        )

        links = []
        if response.status_code == 200:
            attachments = json.loads(response.text)

            # Iterate through the attachments to find PDF files
            for attachment in attachments["results"]:
                if attachment["extensions"]["mediaType"] in supported_types:
                    download_url = attachment["_links"]["download"]
                    links.append(f"{self.__host}/wiki{download_url}")

        return links

    def construct_link(self, **kwargs) -> str:
        if "id" in kwargs and "space" in kwargs:
            id = kwargs["id"]
            space = kwargs["space"]

            return self.__host + "/wiki/spaces/" + space + "/pages/" + id

        return ""

    def download(self, **kwargs) -> list:
        output = []
        try:
            page = self.__api.get_page_by_id(kwargs["id"], expand="body.storage")
            attachments = self.fetch_attachments(kwargs["id"])
            output.append(
                {
                    "space": kwargs["space"],
                    "page": page,
                    "link": kwargs["link"],
                    "title": kwargs["title"],
                    "attachments": attachments,
                }
            )
        except Exception as e:
            print(f"An error occurred: {e}")

        return output
