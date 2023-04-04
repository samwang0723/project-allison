import urllib.request
import io

from project_allison.plugins.plugin_interface import PluginInterface

import urllib.request
from pdfminer.converter import HTMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage


class Pdf(PluginInterface):
    def __init__(self):
        super().__init__()

    def authenticate(self):
        pass

    def download(self, **kwargs) -> list:
        output = []
        try:
            if "url" in kwargs:
                url = kwargs["url"]
                response = urllib.request.urlopen(url)
                pdf_data = response.read()

                # Set up the PDF parser
                resource_manager = PDFResourceManager()
                output_string = io.BytesIO()
                codec = "utf-8"
                laparams = LAParams()

                # Convert the PDF to HTML
                converter = HTMLConverter(
                    resource_manager, output_string, codec=codec, laparams=laparams
                )
                interpreter = PDFPageInterpreter(resource_manager, converter)
                for page in PDFPage.get_pages(io.BytesIO(pdf_data)):
                    interpreter.process_page(page)
                converter.close()

                # Get the HTML output
                html_data = output_string.getvalue().decode()

                output.append(html_data)
        except Exception as e:
            print(f"An error occurred: {e}")

        return output

    def construct_link(self, **kwargs) -> str:
        pass

    def fetch_attachments(self, url) -> list:
        pass
