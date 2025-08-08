from setuptools import setup, find_packages

setup(
    name="civic-ai-classifier",
    version="0.1.0",
    description="Civic infrastructure image classifier",
    packages=find_packages(include=["civic_classifier", "civic_classifier.*"]),
    include_package_data=True,
    package_data={
        "civic_classifier": ["labels.json"],  # make sure labels ship with package
    },
    install_requires=[],
)
