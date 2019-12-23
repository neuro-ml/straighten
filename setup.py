from setuptools import setup, find_packages

from straighten import __version__

classifiers = '''Development Status :: 4 - Beta
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7'''

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

setup(
    name='straighten',
    packages=find_packages(include=('straighten',)),
    version=__version__,
    descriprion='Interpolation along multidimensional curves.',
    license='MIT',
    keywords=[],
    classifiers=classifiers.splitlines(),
    install_requires=requirements,
)
