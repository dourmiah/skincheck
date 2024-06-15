# python .\count_classes_in_dataset.py

import pathlib


def count_files(curr_dir):
    count = 0
    for path in pathlib.Path(curr_dir).iterdir():
        if path.is_file():
            count += 1
    return count


for path in pathlib.Path("../../data_4/train").iterdir():
    if path.is_dir():
        n = count_files(path)
        print(n, path)


# TODO : add a main, manage argument to design intial dir, support recursivity

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Rename all subdirectories to lowercase and replace spaces with underscores."
#     )
#     parser.add_argument(
#         "root_directory",
#         type=str,
#         help="The root directory containing subdirectories to rename",
#     )
#     args = parser.parse_args()

#     rename_subdirectories(args.root_directory)
