import pathlib


def count_files(curr_dir):
    count = 0
    for path in pathlib.Path(curr_dir).iterdir():
        if path.is_file():
            count += 1
    return count


for path in pathlib.Path("./data/train").iterdir():
    if path.is_dir():
        n = count_files(path)
        print(n, path)
