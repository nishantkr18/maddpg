
from setuptools import setup, find_packages

setup(name='maddpg',
      version='0.0.1',
      description='MADDPG-pytorch',
      url='https://github.com/nishantkr18/maddpg',
      author='Igor Mordatch',
      author_email='nniishantkumar@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)