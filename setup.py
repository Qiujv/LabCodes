from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    readme = fh.read()

setup(
    name="labcodes",
    version="0.2",
    author="Qiujv",
    author_email="qiujv@outlook.com",
    description="Codes used in SUSTech superconducting quantum lab.",
    long_description=readme,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "lmfit>=0.9.5",
        # 'h5py>=2.6',
        "tqdm",
        "attrs",
        # 'dpath',
        # 'PyQt5',  # requirements of Labber API here and below.
        "json-numpy",
        "cvxpy",  # for tomography.
        "scikit-learn",  # for state_disc.
        "cmocean",  # for matrix plot.
    ],
)
