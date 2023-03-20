import os
from dotenv import load_dotenv
from atlassian import Confluence


class Wiki:
    def __init__(self):
        load_dotenv()
        self.host = os.environ["CONFLUENCE_HOST"]
        self.username = os.environ["CONFLUENCE_API_USER"]
        self.access_token = os.environ["CONFLUENCE_API_TOKEN"]

    def authenticate(self) -> Confluence:
        confluence = Confluence(
            url=self.host,
            username=self.username,
            password=self.access_token,
            cloud=True,
        )

        return confluence

    def get_link(self, id, space) -> str:
        return self.host + "/wiki/spaces/" + space + "/pages/" + id
