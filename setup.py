from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    README = fh.read()

setup(
    name="Xplique",
    version="0.2.5",
    description="Explanations toolbox for Tensorflow 2",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Thomas FEL",
    author_email="thomas.fel@brown.edu",
    license="MIT",
    install_requires=['tensorflow>=2.1.0', 'numpy', 'scikit-learn', 'scikit-image',
                      'matplotlib', 'scipy'],
    extras_require={
        "tests": ["pytest", "pylint"],
        "docs": ["mkdocs", "mkdocs-material", "numkdoc"],
    },
    packages=find_packages(),
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)