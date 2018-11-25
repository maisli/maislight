#from distutils.core import setup
from setuptools import setup

setup(
        name='maislight',
        version='0.1',
        description='Neuron reconstruction from light microscopy data.',
        url='https://github.com/maisli/maislight',
        author='Lisa Mais',
        author_email='Lisa.Mais@mdc-berlin.de',
        license='MIT',
        packages=[
            'maislight',
            'maislight.gunpowder',
        ]
)
