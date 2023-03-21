import shutil
from rich.console import Console


class DynamicWidthConsole(Console):
    @property
    def width(self):
        columns, _ = shutil.get_terminal_size()
        return columns


# Create a DynamicWidthConsole instance
console = DynamicWidthConsole()
