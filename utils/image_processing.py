
from .misc import get_last_dot_index, tmp_folder
import subprocess
import os
import pytesseract
from PIL import Image as pillow_image


def split_tif(filepath):
    print(filepath)
    command_version = ["convert", "-version"]
    subprocess.run(command_version)

    command = [
        "convert",
        filepath,
        filepath[: get_last_dot_index(filepath)] + "-%02d.tif",
    ]
    command2 = [
        "convert",
        filepath,
        "-crop",
        "100%x100%",
        "+repage",
        "-write",
        filepath[: get_last_dot_index(filepath)] + "-%02d.tif",
        "null:",
    ]

    try:
        subprocess.run(command2, check=True)
        # print("Image splitting completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")

    # list the TIFF individual pages
    page_files = []

    # Get a list of the generated page files
    for file in os.listdir(tmp_folder):
        if file.endswith(".tif"):
            page_files.append(os.path.join(tmp_folder, file))
    # print(page_files)
    # Combine the individual pages into a single PNG file
    cilt_name = os.path.join(tmp_folder, "cilt.png")
    combine_command = ["convert"] + page_files + ["-append", cilt_name]
    try:
        # Execute the command to combine the pages into a PNG file
        subprocess.run(combine_command, check=True)
        # print("Image CILT combining completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image CILT combining failed: {e}")


def extract_bottom(input_path, output_path):
    command = [
        "convert",
        input_path,
        "-gravity",
        "south",
        "-crop",
        "100%x15%",
        output_path,
    ]

    try:
        subprocess.run(command, check=True)
        # print("Bottom extraction completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")


def extract_top(input_path, output_path):
    command = [
        "convert",
        input_path,
        "-gravity",
        "north",
        "-crop",
        "100%x85%",
        output_path,
    ]
    try:
        subprocess.run(command, check=True)
        print("Top extraction completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")


def fix_orientation(file):
    try:
        image = pillow_image.open(file)
        newdata = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"Unable to fix the orientation of the image: {e}")
        return
    if newdata["rotate"]:
        command = [
            "convert",
            file,
            "-rotate",
            str(newdata["rotate"]),
            file,
        ]
        try:
            subprocess.run(command, check=True)
            print("Page rotated successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Page rotation failed: {e}")


def split_into_pages_top_and_bottom(filepath, lo_page, hi_page):
    # iterate over pages
    last_dot_index = get_last_dot_index(filepath)
    files = os.listdir(tmp_folder)
    files = [file for file in files if "-" and ".tif" in file]
    files.sort()
    for file in files[lo_page: len(files) if hi_page == -1 else hi_page]:
        bottom_filename = os.path.join(
            tmp_folder, f"{file[:last_dot_index]}-bottom{file[last_dot_index:]}"
        )
        print(bottom_filename)
        top_filename = os.path.join(
            tmp_folder, f"{file[:last_dot_index]}-top{file[last_dot_index:]}"
        )
        print(top_filename)
        
        if os.path.isfile(bottom_filename) and os.path.isfile(top_filename):
            continue
        
        page_filepath = os.path.join(tmp_folder, file)
        fix_orientation(page_filepath)
        print("Splitting the following page into the two files below:")
        extract_bottom(page_filepath, bottom_filename)
        extract_top(page_filepath, top_filename)


def add_border(filepath):
    input_file = filepath
    output_file = filepath[: get_last_dot_index(filepath)] + "-border.png"

    command = [
        "convert",
        input_file,
        "-bordercolor",
        "lime",
        "-border",
        "5x5",
        output_file,
    ]
    try:
        subprocess.run(command, check=True)
        # print("Border addition completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")


def recombine(top_path, bottom_path):
    # add border to both top and bottom image
    add_border(top_path)
    add_border(bottom_path)

    # append bottom to top and compress as png
    command = [
        "convert",
        top_path[: get_last_dot_index(top_path)] + "-border.png",
        bottom_path[: get_last_dot_index(bottom_path)] + "-border.png",
        "-append",
        "-define",
        "png:compression-filter=5",
        "-define",
        "png:compression-level=9",
        "-define",
        "png:compression-strategy=1",
        os.path.join(tmp_folder, "image-for-A2I.png"),
    ]
    try:
        subprocess.run(command, check=True)
        # print("Image append completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")

    os.remove(
        top_path[: get_last_dot_index(top_path)] + "-border.png"
    )  # Remove the intermediate files
    os.remove(bottom_path[: get_last_dot_index(bottom_path)] + "-border.png")
    