from setuptools import setup

setup(
    name='micronet',
    version='0.1',
    description='CNN setup for microtubule prediction',
    url='https://github.com/nilsec/micronet',
    author='Nils Eckstein',
    author_email='ecksteinn@janelia.org',
    license='MIT',
    packages=[
        'micronet',
            ],
    install_requires = [
        'numpy',
        'scipy',
            ],
)   
