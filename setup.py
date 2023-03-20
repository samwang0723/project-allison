from setuptools import setup, find_packages

import jarvis


def long_description():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt", encoding="utf-8") as f:
        return f.read().splitlines()


setup(
    name="jarvis",
    version=jarvis.__version__,
    packages=find_packages(),
    install_requires=get_requirements(),
    entry_points={
        "console_scripts": [
            "jarvis=jarvis.__main__:main",
        ],
    },
    author=jarvis.__author__,
    author_email=jarvis.__email__,
    description=jarvis.__description__,
    long_description=long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/samwang0723/openai-training",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Environment :: Console",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
