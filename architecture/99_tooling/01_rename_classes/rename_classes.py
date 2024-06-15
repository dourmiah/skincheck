#  python .\rename_dataset.py ../../data_24
import os
import argparse


def rename_subdirectories(root_dir):
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for dir_name in dirs:
            new_name = dir_name.lower().replace("photos", "").replace(" ", "_")
            old_path = os.path.join(root, dir_name)
            new_path = os.path.join(root, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                # print(f'Renamed: "{old_path}" to "{new_path}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename all subdirectories to lowercase and replace spaces with underscores."
    )
    parser.add_argument(
        "root_directory",
        type=str,
        help="The root directory containing subdirectories to rename",
    )
    args = parser.parse_args()

    rename_subdirectories(args.root_directory)
