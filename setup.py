from setuptools import find_packages, setup
from typing import List

EXTRA_COMMAND = "-e ."


def get_requirements(filepath:str) -> List[str]:
    requirements = []
    with open(filepath) as file:
        requirements = file.readline()
        requirements = [package.replace("\n", " ") for package in requirements]

        if EXTRA_COMMAND in requirements:
            requirements.remove(EXTRA_COMMAND)
    return requirements



setup(
    name = "student_performance",
    version = '0.0.1',
    author = 'Mahi',
    author_email = "mahichouhan005@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)