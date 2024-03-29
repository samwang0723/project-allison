import os
import csv
import inspect

from chromadb.api.models.Collection import Collection

from project_allison.plugins.plugin_interface import PluginInterface

from .constants import SOURCE_FILE

__plugins = {}
__plugin_mapping = {
    "Web": "WEB",
    # "Drive": "GOOGLE",
    "Wiki": "OTHERS",
    "NewsAPI": "NEWS",
    "Finance": "FINANCE",
    "Pdf": "PDF",
    "Jira": "JIRA",
}


def load_plugins():
    # Assuming the current working directory is the root of your project
    folder_path = "project_allison/plugins"
    skip_files = ["__init__.py", "plugin_interface.py"]

    # Iterate over all the files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a Python module
        if file_name.endswith(".py") and file_name not in skip_files:
            # Import the module
            module_name = file_name[:-3]  # Remove the .py extension
            module = __import__(f"{folder_path}.{module_name}", fromlist=["*"])

            # Iterate over all the objects in the module
            for obj_name in dir(module):
                obj = getattr(module, obj_name)
                # Check if the object is a class and has the desired attributes
                if (
                    obj_name != "PluginInterface"
                    and inspect.isclass(obj)
                    and issubclass(obj, PluginInterface)
                ):
                    # Do something with the class
                    print(f"load class {obj_name} in module {module_name}")
                    if obj_name in __plugin_mapping:
                        __plugins[__plugin_mapping[obj_name]] = obj()


def get_plugin(type_id: str) -> PluginInterface:
    return __plugins[type_id]


def download_content(download_type: str, vdb=None):
    __do_authenticate()

    if download_type == "source":
        return __download_from_source_csv(vdb)
    elif download_type == "gmail":
        executer = __plugins["GOOGLE"]
        return executer.download(file_type="gmail")
    elif download_type == "news":
        executer = __plugins["NEWS"]
        return executer.download()
    elif download_type == "finance::picked":
        executer = __plugins["FINANCE"]
        return executer.download(type="picked_stocks")


def __download_from_source_csv(vdb: Collection) -> list[str]:
    pages = []

    with open(SOURCE_FILE) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            space = row["space"]
            id = row["page_id"]
            header = row["type"]

            if space in __plugins:
                executer = __plugins[space]
            else:
                executer = __plugins["OTHERS"]

            if space == "GOOGLE":
                link = executer.construct_link(id=id)
                if __not_indexed(vdb, link):
                    print(f" > Downloading {link}, space: {space}, id: {id}")
                    page = executer.download(file_type="gdrive", file_id=id)
                    pages.append(
                        {"space": space, "page": page[0], "link": link, "title": header}
                    )
            elif space == "WEB":
                link = id
                if __not_indexed(vdb, link):
                    print(f" > Downloading {link}, space: {space}, id: {id}")
                    raw_data = executer.download(url=id)
                    for page in raw_data:
                        attachments = executer.fetch_attachments(page)
                        pages.append(
                            {
                                "space": space,
                                "page": page,
                                "link": link,
                                "attachments": attachments,
                                "title": header,
                            }
                        )
            elif space == "PDF":
                link = id
                if __not_indexed(vdb, link):
                    print(f" > Downloading {link}, space: {space}, id: {id}")
                    raw_data = executer.download(url=id)
                    for page in raw_data:
                        pages.append(
                            {
                                "space": space,
                                "page": page,
                                "link": link,
                                "title": header,
                            }
                        )
            else:
                link = executer.construct_link(id=id, space=space)
                if __not_indexed(vdb, link):
                    print(f" > Downloading {link}, space: {space}, id: {id}")

                    data = executer.download(
                        id=id, space=space, link=link, title=header
                    )
                    if len(data) > 0:
                        pages.append(data[0])

    return pages


def __do_authenticate():
    for plugin in __plugins.values():
        plugin.authenticate()


def __not_indexed(vdb, link) -> bool:
    get_where_result = vdb.get(
        where={"link": link},
    )

    # Manual refresh content
    # if len(get_where_result["ids"]) > 0:
    #    __debug_documents(vdb, get_where_result["ids"])
    #    if "d3ecf9d9-2a92-4682-b110-51e80b939d68" in get_where_result["ids"]:
    #        vdb.delete(ids=get_where_result["ids"])
    #        return True
    #    return False

    return len(get_where_result["ids"]) == 0


def __debug_documents(vdb, result_ids):
    if len(result_ids) > 0:
        get_ids_result = vdb.get(
            ids=result_ids,
        )
        print("\nGET ids: ", result_ids)
        print("\nDocuments: \n", get_ids_result["documents"])
