import setuptools
from setuptools import find_packages

setuptools.setup(    
    name='HTorch',
    version='0.1',
    description='HTorch',
    author='',
    url='https://github.com/ydtydr/HTorch',
    project_urls={
        "Bug Tracker": "https://github.com/ydtydr/HTorch",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
)