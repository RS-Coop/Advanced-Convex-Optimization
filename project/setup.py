from setuptools import setup, find_packages

setup(
    name='core',
    version='0.0.1',
    packages = ['core'],
    include_package_data=True,
    install_requires=['scipy', 'numpy', 'torch', 'torchvision']
)
