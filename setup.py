from setuptools import setup, find_packages

setup(
    name='transitforecast',
    version='0.0.0',
    author='Benjamin V. Rackham',
    author_email='brackham@mit.edu',
    description='Probabilistic forecasting of candidate exoplanetary transits',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/disruptiveplanets/transitforecast/',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().split('\n'),
)
