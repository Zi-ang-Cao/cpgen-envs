# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()
    long_description = "".join(lines)

setup(
    name="cpgen_envs",
    packages=find_packages(),
    install_requires=[
        "robosuite>=1.5.0",
        "robosuite_task_zoo @ git+https://github.com/kevin-thankyou-lin/robosuite-task-zoo.git"
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3.8",
    description="CP-Gen Environments",
    author="Kevin Lin",
    url="https://github.com/kevin-thankyou-lin/cpgen-envs",
    author_email="kevin.lin20000@gmail.com",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
