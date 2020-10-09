from setuptools import Extension, setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="melgan",  # Replace with your own username
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'unidecode',
        'inflect',
        'scipy',
        'librosa',
        'Cython',
        'pillow',
        'pyyaml',
        'tqdm',
        'tensorboardX',
        'matplotlib'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
