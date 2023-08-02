# setup.py

from setuptools import setup, find_packages

setup(
    name='your_app_name',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'plotly',
        'pycaret',
        'pandas',
        'pandas-profiling',
        'streamlit-pandas-profiling'
    ],
)
