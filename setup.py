import os

from setuptools import find_packages, setup

from grasp_ldm import __version__

here = os.path.abspath(os.path.dirname(__file__))
requires_list = []
with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # Skip comments, empty lines, and editable/index flags
        if line and not line.startswith("#") and not line.startswith("-e"):
            requires_list.append(line)

setup(
    name="grasp_ldm",
    version=__version__,
    author="Kuldeep Barad",
    packages=find_packages(exclude=["tests*", "tools*", "data_utils*"]),
    python_requires=">=3.8.0",
    install_requires=requires_list,
)
