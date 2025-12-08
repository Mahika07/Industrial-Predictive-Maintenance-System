from setuptools import setup, find_packages

setup(
    name="predictive_maintenance_ai4i",
    version="1.0.0",
    author="Mahika Sharma",
    description="Machine Learning project for Predictive Maintenance using the AI4I 2020 dataset.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "xgboost",
        "imbalanced-learn",
        "joblib"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
