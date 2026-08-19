# setup.py

from setuptools import setup, find_packages

setup(
    name="gannddi",
    version="0.1.0",
    description="Gated Attention Network for Drug-Drug Interaction Prediction",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "rdkit-pypi>=2022.3.5",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "click>=8.0.0",
        "tensorboard>=2.7.0"
    ],
    python_requires=">=3.8",
)