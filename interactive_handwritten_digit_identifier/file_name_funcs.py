"""A module containing a function for getting valid save names in folders."""
from os import listdir

def get_valid_file_name(save_path: str, format_start: str = "", format_end: str = "", file_extension: str = ".txt"):
    """Returns a valid file name within a given save path. Can use a custom format, and
    allows one index number to be included.

    Parameters:
        - save_path: str folder name.
        - format_start: Optional str file name part before index number. Default value is an empty string
        - format_end: Optional str file name part after index number. Default value is an empty string.
        - file_extension: Optional str file extension including dot ('.'). Default value is '.txt'.

    Format info:
        - Default format is '{index}.txt'"""

    dir_files = listdir(save_path)

    indx = 0
    found_valid_index = False
    while not found_valid_index:
        if f"{format_start}{indx}{format_end}{file_extension}" in dir_files:
            indx += 1
        else:
            found_valid_index = True
    name = f"{format_start}{indx}{format_end}{file_extension}"
    file_name = f"{save_path}\\{name}"

    return file_name
