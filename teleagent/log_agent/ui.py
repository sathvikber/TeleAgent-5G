import tkinter as tk

from tkinter import (
    filedialog,
    messagebox
)


SUPPORTED_FILES = [
    (
        "Supported Files",
        "*.pdf *.docx *.txt *.csv "
        "*.json *.xlsx *.xls"
    ),

    ("PDF", "*.pdf"),
    ("Word", "*.docx"),
    ("Text", "*.txt"),
    ("CSV", "*.csv"),
    ("JSON", "*.json"),
    ("Excel", "*.xlsx *.xls"),

    ("All Files", "*.*")
]


def select_log_files():

    root = tk.Tk()

    root.withdraw()

    root.attributes(
        "-topmost",
        True
    )

    files = filedialog.askopenfilenames(
        title="Select 5G Network Logs",
        filetypes=SUPPORTED_FILES
    )

    root.destroy()

    return list(files)


def show_success(message):

    root = tk.Tk()

    root.withdraw()

    root.attributes(
        "-topmost",
        True
    )

    messagebox.showinfo(
        "5G Log Agent",
        message
    )

    root.destroy()