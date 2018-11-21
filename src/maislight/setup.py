from distutils.core import setup

setup(
        name='maislight',
        version='0.1',
        description='Neuron reconstruction from light.',
        url='https://github.com/maisli/maislight',
        author='Lisa Mais',
        author_email='maisl@janelia.hhmi.org',
        license='MIT',
        packages=[
            'maislight',
            'maislight.gunpowder',
        ]
)
