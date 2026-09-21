"""Install torchtext compatibility shim for scGPT on PyTorch 2.5+."""

from setuptools import setup, find_packages

setup(
    name="torchtext",
    version="0.0.1",
    description="Minimal torchtext shim providing Vocab class for scGPT compatibility",
    packages=find_packages(),
    python_requires=">=3.9",
)
