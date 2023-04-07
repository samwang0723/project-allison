import os

from project_allison.plugins.plugin_interface import PluginInterface

from jira import JIRA


class Jira(PluginInterface):
    def __init__(self):
        super().__init__()

        self.__host = os.environ["CONFLUENCE_HOST"]
        self.__username = os.environ["CONFLUENCE_API_USER"]
        self.__access_token = os.environ["CONFLUENCE_API_TOKEN"]
        self.__api = None
        self.__auth = (self.__username, self.__access_token)

    def authenticate(self):
        self.__api = JIRA(
            server=self.__host,
            basic_auth=self.__auth,
        )

    def fetch_attachments(self, data) -> list:
        pass

    def construct_link(self, **kwargs) -> str:
        return f"{self.__host}/browse/{kwargs['id']}"

    def download(self, **kwargs) -> list:
        output = []
        try:
            parent_issue = self.__api.issue(kwargs["id"])
            output.append(parent_issue)

            # get the child issues
            child_issues = self.__api.search_issues("parent = " + parent_issue.key)
            for child in child_issues:
                output.append(child)

            epic_child_issues = self.__api.search_issues(
                "parentEpic = " + parent_issue.key
            )
            for epic_child in epic_child_issues:
                output.append(epic_child)

        except Exception as e:
            print(f"An error occurred: {e}")

        return output
