from setuptools import setup, find_packages

setup(
    name="pyfooda",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "openpyxl>=3.0.0",  # for Excel support
    ],
    author="Jerome",
    author_email="your.email@example.com",  # Update this
    description="A Python API for accessing USDA FoodData Central data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pyfooda",  # Update this
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    include_package_data=True,
) 