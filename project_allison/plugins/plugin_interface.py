from project_allison.constants import ENV_PATH
from dotenv import load_dotenv
from abc import ABC, abstractmethod


class PluginInterface(ABC):
    def __init__(self):
        load_dotenv(dotenv_path=ENV_PATH)

    @abstractmethod
    def authenticate(self):
        pass

    @abstractmethod
    def download(self, **kwargs) -> list:
        pass

    @abstractmethod
    def construct_link(self, **kwargs) -> str:
        pass

    @abstractmethod
    def fetch_attachments(self, data) -> list:
        pass
