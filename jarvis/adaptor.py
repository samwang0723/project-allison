import os
import csv
from .googleapis import Drive
from .confluence import Wiki
from rich.progress import Progress

MATERIAL_FILE = "../data/material.csv"
SOURCE_FILE = "../data/source.csv"


def download_content(with_gdrive: bool = False, with_confluence: bool = False):
    if with_gdrive:
        google_drive = Drive()
        google_drive.authenticate()

    if with_confluence:
        confluence_wiki = Wiki()
        confluence_wiki.authenticate()

    pages = []
    downloaded = __get_previous_downloaded()

    with open(SOURCE_FILE) as csvfile:
        reader = csv.reader(csvfile)
        num_rows = sum(1 for _ in reader)

    with Progress() as progress:
        task = progress.add_task("[cyan]Download missing content", total=num_rows)
        progress.update(task, advance=1)

        with open(SOURCE_FILE) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                space = row["space"]
                id = row["page_id"]

                if space == "GOOGLE":
                    link = google_drive.get_link(id)
                    if link not in downloaded and google_drive != None:
                        page = google_drive.download_file(id)
                        pages.append({"space": space, "page": page, "link": link})
                else:
                    link = confluence_wiki.get_link(id, space)
                    if link not in downloaded:
                        page = confluence_wiki.get_page_by_id(id, expand="body.storage")
                        pages.append({"space": space, "page": page, "link": link})

                progress.update(task, advance=1)

    return pages


def __get_previous_downloaded() -> list[str]:
    downloaded = []
    if os.path.isfile(MATERIAL_FILE) is False:
        return []

    with open(MATERIAL_FILE) as csvfile:
        reader = csv.reader(csvfile)
        num_rows = sum(1 for _ in reader)

    with Progress() as progress:
        task = progress.add_task("[cyan]Retrieve indexed material", total=num_rows)
        progress.update(task, advance=1)
        with open(MATERIAL_FILE) as downloaded_file:
            downloaded_reader = csv.DictReader(downloaded_file)
            for d in downloaded_reader:
                downloaded.append(d["link"])
                progress.update(task, advance=1)

    return list(set(downloaded))
