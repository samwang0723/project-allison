import os
import requests
import re

from jarvis.plugins.plugin_interface import PluginInterface


class Web(PluginInterface):
    def __init__(self):
        super().__init__()
        if os.environ["SKIP_SSL_VERIFICATION"] == "1":
            self.verify_ssl = False
        else:
            self.verify_ssl = True

    def authenticate(self):
        pass

    def download(self, **kwargs) -> list:
        output = []
        if "url" in kwargs:
            url = kwargs["url"]
            response = requests.get(url, verify=self.verify_ssl)
            if response.status_code == 200:
                output.append(response.text)

        return output

    def construct_link(self, **kwargs) -> str:
        pass

    def fetch_attachments(self, data) -> list:
        attachments = []
        regex = r'/href=["\'][^"\']*?\.(png|jpe?g|gif)(?:\?[^"\']*)?["\']/g'
        for match in re.finditer(regex, data):
            attachments.append(match.group(1))

        return attachments
