from setuptools import find_packages, setup

setup(
    name="grasp_ldm",
    version="0.0.1",
    author="Kuldeep Barad",
    packages=find_packages(exclude=["tests*", "tools*", "data_utils*"]),
    python_requires=">=3.9",
)
