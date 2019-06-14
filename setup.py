import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DeepProteinPred",
    version="0.0.3",
    author="Jin Li",
    author_email="jinli7255@gmail.com",
    description="Predict and Generate Protein Structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JinLi711/Protein-Structures",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)