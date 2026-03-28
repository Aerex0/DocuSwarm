from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="docuswarm-financial-qa",
    version="1.0.0",
    author="DocuSwarm Contributors",
    author_email="team@example.com",
    description="Multi-Agent QA System for Financial Documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aerex0/DocuSwarm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-asyncio>=0.24.0",
            "pytest-cov>=6.0.0",
            "black>=24.0.0",
            "ruff>=0.7.0",
            "mypy>=1.13.0",
            "pre-commit>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "financial-qa=pipeline.orchestrator:main",
            "preprocess-docs=scripts.preprocess_documents:main",
            "setup-chromadb=scripts.setup_chromadb:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "*.md"],
    },
)
