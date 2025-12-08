from setuptools import setup, find_packages

HYPEN_E_DOT = '-e .'
def get_requirements(file_path):
    """
    Reads a requirements.txt file and returns a list of dependencies.
    Ignores empty lines and comments.
    """
    requirements = []
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[req.replace("\n","") for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="predictive_maintenance_ai4i",
    version="1.0.0",
    author="Mahika Sharma",
    description="Machine Learning project for Predictive Maintenance using the AI4I 2020 dataset.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_requirements('requirements.txt')
    
)
