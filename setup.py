from setuptools import setup, find_packages
import codecs
import os


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='moseq2-viz',
    author='Datta Lab',
    description='To boldly go where no mouse has gone before',
    version=get_version("moseq2_viz/__init__.py"),
    packages=find_packages(),
    platforms=['mac', 'unix'],
    install_requires=['tqdm==4.40.0', 'matplotlib==3.1.2', 'click==7.0', 'dtaidistance==1.2.3', 'scikit-learn==0.20.3',
                      'ruamel.yaml==0.16.5', 'seaborn==0.11.0', 'opencv-python==4.1.2.30', 'psutil==5.6.7',
                      'pandas==1.0.5', 'networkx==2.4', 'numpy==1.18.3', 'h5py==2.10.0', 'cytoolz==0.10.1',
                      'joblib==0.15.1', 'scipy==1.3.2', 'statsmodels==0.10.2'],
    python_requires='>=3.6,<3.8',
    entry_points={'console_scripts': ['moseq2-viz = moseq2_viz.cli:cli']}
)
