import os
import csv

from jarvis.repository.confluence import Wiki
from jarvis.repository.googleapis import Drive
from jarvis.repository.web import Web
from jarvis.repository.news_api import NewsAPI

from .constants import SOURCE_FILE, MATERIAL_FILE


def download_gmail():
    google_drive = Drive()
    google_drive.authenticate()

    return google_drive.download(file_type="gmail")


def download_news():
    news_api = NewsAPI()

    return news_api.download()


def download_content(
    with_gdrive: bool = True, with_confluence: bool = False, with_web: bool = False
):
    if with_gdrive:
        google_drive = Drive()
        google_drive.authenticate()

    if with_confluence:
        confluence_wiki = Wiki()
        confluence_wiki.authenticate()

    if with_web:
        web = Web()

    pages = []
    downloaded = __get_previous_downloaded()

    with open(SOURCE_FILE) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            space = row["space"]
            id = row["page_id"]

            if space == "GOOGLE":
                link = google_drive.construct_link(id=id)
                if link not in downloaded and with_gdrive:
                    print(f" > Downloading {link}, space: {space}, id: {id}")
                    page = google_drive.download(file_type="gdrive", file_id=id)
                    pages.append({"space": space, "page": page[0], "link": link})
            elif space == "WEB":
                link = id
                if link not in downloaded and with_web:
                    print(f" > Downloading {link}, space: {space}, id: {id}")
                    raw_data = web.download(url=id)
                    for page in raw_data:
                        attachments = web.fetch_attachments(page)
                        pages.append(
                            {
                                "space": space,
                                "page": page,
                                "link": link,
                                "attachments": attachments,
                            }
                        )
            else:
                link = confluence_wiki.construct_link(id=id, space=space)
                if link not in downloaded:
                    print(f" > Downloading {link}, space: {space}, id: {id}")

                    data = confluence_wiki.download(id=id, space=space, link=link)
                    if len(data) > 0:
                        pages.append(data[0])

    return pages


def __get_previous_downloaded() -> list[str]:
    downloaded = []
    if os.path.isfile(MATERIAL_FILE) is False:
        return []

    with open(MATERIAL_FILE) as downloaded_file:
        downloaded_reader = csv.DictReader(downloaded_file)
        for d in downloaded_reader:
            downloaded.append(d["link"])

    return list(set(downloaded))
