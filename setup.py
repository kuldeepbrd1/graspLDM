import os

from setuptools import find_namespace_packages, find_packages, setup

from grasp_ldm import __version__

# here = os.path.abspath(os.path.dirname(__file__))
# requires_list = []
# with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
#     for line in f:
#         requires_list.append(str(line))

setup(
    name="grasp_ldm",
    version=__version__,
    author="Kuldeep Barad",
    # TODO: Improve grasp_ldm_utils module by combining internal and external utils
    packages=["grasp_ldm", "grasp_ldm.tools", "grasp_ldm_utils"],
    # packages=find_packages(),
    package_dir={
        "grasp_ldm": "grasp_ldm",
        "grasp_ldm.tools": "tools",
        "grasp_ldm_utils": "utils",
    },
    python_requires=">=3.8.0, <3.10",
    # install_requires=requires_list,
)
