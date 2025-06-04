from setuptools import setup, find_packages

setup(
    name="ml_solubility",
    version="0.1.0",
    packages=find_packages(),  # This will include the top-level 'src' directory
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "torch",
        "mlflow",
        "pyyaml",
        "joblib",
    ],
    entry_points={
        "console_scripts": [
            "ml-solubility=src.run:main",  # Optional CLI entry point
        ],
    },
)