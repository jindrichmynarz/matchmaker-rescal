from setuptools import setup

setup(
    name = "matchmaker_rescal",
    description = "Evaluation runner for the RESCAL-based matchmakers",
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    url = "https://github.com/jindrichmynarz/matchmaker-rescal",
    author = "Jindřich Mynarz",
    author_email = "mynarzjindrich@gmail.com",
    license = "Eclipse 1.0",
    version = "0.1",
    packages = ["matchmaker_rescal"],
    install_requires = [
        "edn_format",
        "numpy",
        "scipy"
    ],
    dependency_links = [
        "https://github.com/mnick/rescal.py.git#egg=rescal-0.1"
    ]
)
