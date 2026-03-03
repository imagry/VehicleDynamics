"""Setup script for simulation repository."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="vehicle-simulation",
    version="1.0.0",
    description="Vehicle dynamics simulation and parameter fitting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/vehicle-simulation",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=2.0.0",
        "pandas>=1.3.0",
        "boto3>=1.20.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fetch-trips=scripts.fetch_trips:main",
            "parse-trips=scripts.parse_trips:main",
            "fit-params=scripts.fit_params:main",
            "simulate-trip=scripts.simulate_trip:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
)
