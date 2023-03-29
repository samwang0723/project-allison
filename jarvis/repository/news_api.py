import os

from jarvis.constants import ENV_PATH
from newsapi import NewsApiClient
from dotenv import load_dotenv


class NewsAPI:
    def __init__(self):
        load_dotenv(dotenv_path=ENV_PATH)
        self.api_key = os.environ["NEWS_API_KEY"]
        self.newsapi = NewsApiClient(api_key=self.api_key)

    def download_file(self):
        output = []
        try:
            data = self.newsapi.get_top_headlines(
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
