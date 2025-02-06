from setuptools import setup, find_packages

setup(
    name="sigmaeval",
    version="0.1.0",
    author="Sebastian Haan",
    description="Evaluation metrics for probabilistic and quantile predictions including CRPS, Brier score, and other uncertainty quantification metrics.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sebhaan/sigmaeval",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)