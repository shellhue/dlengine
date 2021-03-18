import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dlengine", # Replace with your own username
    version="0.0.6",
    author="zeyu huang",
    author_email="zeyu_huang@163.com",
    description="dlengine is a trainning engine targeting to separating miscellaneous trainning things from modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)