import os
import requests

from project_allison.plugins.plugin_interface import PluginInterface


class Finance(PluginInterface):
    def __init__(self):
        super().__init__()
        self.__host = os.environ["STOCK_API_HOST"]

    def authenticate(self):
        pass

    def download(self, **kwargs) -> list:
        output = []
        try:
            if "type" in kwargs:
                type = kwargs["type"]
                if type == "picked_stocks":
                    output = self.__get_picked_stocks()
        except Exception as e:
            print(f"An error occurred: {e}")

        return output

    def construct_link(self, **kwargs) -> str:
        pass

    def fetch_attachments(self, data) -> list:
        pass

    def __get_picked_stocks(self):
        url = f"{self.__host}/v1/pickedstocks"
        response = requests.get(url)

        if response.status_code == 200:
            return self.__parse_stock_data(response.json())
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

    def __parse_stock_data(self, data) -> list:
        entries = data["entries"]
        result = []
        for entry in entries:
            stock_id = entry["stockID"]
            name = entry["name"]
            date = entry["date"]
            concentration1 = entry["concentration1"]
            volume = entry["volume"]
            foreign = entry["foreign"]
            trust = entry["trust"]
            close_price = entry["close"]
            diff = entry["diff"]
            quote_change = entry["quoteChange"]

            result.append(
                [
                    date,
                    stock_id,
                    name,
                    volume,
                    close_price,
                    diff,
                    f"{quote_change}%",
                    concentration1,
                    foreign,
                    trust,
                ],
            )
        return result
