import os

from project_allison.plugins.plugin_interface import PluginInterface

from newsapi import NewsApiClient


class NewsAPI(PluginInterface):
    def __init__(self):
        super().__init__()
        self.__api_key = os.environ["NEWS_API_KEY"]
        self.__newsapi = NewsApiClient(api_key=self.__api_key)

    def authenticate(self):
        pass

    def download(self, **kwargs):
        output = []
        try:
            data = self.__newsapi.get_top_headlines(
                q="bitcoin", category="business", language="en", country="us"
            )

            # Access data
            if "status" in data and data["status"] == "ok":
                articles = data["articles"]

                for article in articles:
                    output.append(
                        {
                            "link": article["url"],
                            "body": f"Subject: {article['title']} ({article['author']}, {article['publishedAt']}) \n {article['description']}",
                        }
                    )
        except Exception as e:
            print(f"Error: {e}")

        return output

    def construct_link(self, **kwargs) -> str:
        pass

    def fetch_attachments(self, data) -> list:
        pass
