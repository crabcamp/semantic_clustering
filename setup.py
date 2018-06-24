from pathlib import Path

from setuptools import find_packages, setup

requirements = Path(__file__).parent / 'requirements/core.txt'

with requirements.open(mode='rt', encoding='utf-8') as fp:
    install_requires = [line.strip() for line in fp]

readme = Path(__file__).parent / 'README.rst'

with readme.open(mode='rt', encoding='utf-8') as fp:
    readme_text = fp.read()

setup(
    name='keywords_similarity',
    version='0.0.1',
    description='Semantic clustering of keywords groups.',
    long_description=readme_text,
    license='MIT',
    author='Luka Shostenko',
    author_email='luka.shostenko@gmail.com',
    url='https://github.com/LShostenko/keywords_similarity/',
    packages=find_packages(include=['keywords_similarity.*']),
    py_modules=['keywords_similarity.match'],
    python_requires='>=3.5.0',
    install_requires=install_requires,
    include_package_data=True,
)
