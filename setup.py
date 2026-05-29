from setuptools import setup, find_packages

setup(
    name             = "dlpmi",
    version          = "1.1.0",
    description      = (
        "DLPMI: A Differentiable Framework for Lumped Parameter Model "
        "Inversion and Uncertainty Quantification in Groundwater Age Dating"
    ),
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    author           = "DLPMI Development Team",
    license          = "MIT",
    packages         = find_packages(),
    python_requires  = ">=3.9",
    install_requires = [
        "torch>=2.0",
        "numpy>=1.23",
        "scipy>=1.9",
    ],
    extras_require = {
        "plot": ["matplotlib>=3.6"],
        "dev":  ["pytest", "pytest-cov"],
    },
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
