import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("requirements.txt", "r") as fh:
    requires = [line for line in fh.read().splitlines() if line != ""]

setuptools.setup(
    name="ao3-api",
    version="2.4.0",
    author="Joseph Burris",
    author_email="jtrainrva@gmail.com",
    description="Train's Dice Stats Package",
    python_requires='>=3.8',
    install_requires=requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jtrainrva/dice_roll",
    packages=setuptools.find_packages(),
    keywords=['python','dice'],
    classifiers=[
        "Programming Language :: Python :: 3"
        "Operating System :: OS Independent",
    ],
)
