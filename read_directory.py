from pathlib import Path


def load_dataset(path):
    full_paths = list(path.iterdir())
    return full_paths


def load_original_file(path):
    original_file_path_directory = str(path) + "/original"
    original_file_path_file = Path(original_file_path_directory)
    original_file = list(original_file_path_file.iterdir())

    return original_file[0]


def load_plagiarized_files(path):
    plagiarized_path = str(path) + "/plagiarized"
    plagiarized_path = Path(plagiarized_path)
    plagiarized_L_directories = list(plagiarized_path.iterdir())
    plagiarized_files_paths = []
    for path in plagiarized_L_directories:
        plagiarized_files_paths = plagiarized_files_paths + list(path.iterdir())

    plagiarized_files = []
    for path in plagiarized_files_paths:
        plagiarized_files = plagiarized_files + list(path.iterdir())

    return plagiarized_files


def load_non_plagiarized_files(path):
    non_plagiarized_path = str(path) + "/non-plagiarized"
    non_plagiarized_path = Path(non_plagiarized_path)

    non_plagiarized_inner = list(non_plagiarized_path.iterdir())

    non_plagiarized_files = []
    for path in non_plagiarized_inner:
        non_plagiarized_files = non_plagiarized_files + list(path.iterdir())

    return non_plagiarized_files


path = Path("/home/stormblessed/parser-plagio/data/IR-Plag-Dataset/case-01")

load_non_plagiarized_files(path)
