import os


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def data_dir(*parts: str) -> str:
    return os.path.join(repo_root(), "data", *parts)


def processed_path(filename: str) -> str:
    return os.path.join(data_dir("processed"), filename)


def raw_path(*parts: str) -> str:
    return os.path.join(data_dir("raw"), *parts)
