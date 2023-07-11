
tmp_folder = "tmp"

def get_last_dot_index(filepath):
    last_dot_index = filepath[::-1].find(".")
    return -last_dot_index - 1 if last_dot_index != -1 else 0