from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'


def get_requrements(file_path: str) -> List[str]:
    requrements = []
    with open(file_path) as file_obj:
        requrements = file_obj.readlines()
        requrements = [req.replace("\n", "") for req in requrements]

        if HYPEN_E_DOT in requrements:
            requrements.remove(HYPEN_E_DOT)
    return requrements


setup(
    name='Used-Cars-Price-Prediction',
    version='1.0.0',
    author='Akhil Raj',
    author_email='akhil_raj@outlook.com',
    packages=find_packages(),
    install_requires=get_requrements('requrements.txt')
)
