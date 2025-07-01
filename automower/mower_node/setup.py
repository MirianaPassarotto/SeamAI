from setuptools import setup
import os
from glob import glob

package_name = "mower_node"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    include_package_data=True,
    package_data={
        "": ["templates/*.html"],
    },
    zip_safe=True,
    maintainer="Johannes",
    maintainer_email="johannesferm1@gmail.com",
    description="Node for collecting data, testing the model and displaying a map",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mower_node = mower_node.mower_node:main"
        ],
    },
)
