from setuptools import setup, find_packages

def get_version():
    with open('pyfooda/VERSION') as f:
        return f.read().strip()

setup(
    name="pyfooda",
    version=get_version(),
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "rank-bm25>=0.2.2",
    ],
    author="Jerome",
    description="Python API for USDA FoodData Central with LLM-powered food aggregation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    include_package_data=True,
) 