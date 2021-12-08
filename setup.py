import os
from pathlib import Path

from setuptools import find_packages, setup

THIS_DIR = Path(__file__).parent
DATA_ROOT = 'encodechka_eval/data'


def _load_requirements(path_dir: Path, comment_char: str = "#"):
    with open(os.path.join(path_dir, "requirements.txt"), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    requirements = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)]
        if ln:  # if requirement is not empty
            requirements.append(ln)
    return requirements


setup(
    name="encodechka_eval",
    version="0.0.1",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=_load_requirements(THIS_DIR),
    data_files=[(DATA_ROOT, [os.path.join(DATA_ROOT, f) for f in os.listdir(DATA_ROOT)])],
)
