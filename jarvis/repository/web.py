import os
import requests
import re

from jarvis.constants import ENV_PATH

from dotenv import load_dotenv


class Web:
    def __init__(self):
        load_dotenv(dotenv_path=ENV_PATH)
        if os.environ["SKIP_SSL_VERIFICATION"] == "1":
            self.verify_ssl = False
        else:
            self.verify_ssl = True

    def download_file(self, url: str):
        response = requests.get(url, verify=self.verify_ssl)
        if response.status_code == 200:
            return response.text
        else:
            return None

    def get_attachments(self, raw_content):
        attachments = []
        regex = r'/href=["\'][^"\']*?\.(png|jpe?g|gif)(?:\?[^"\']*)?["\']/g'
        for match in re.finditer(regex, raw_content):
            attachments.append(match.group(1))

        return attachments
