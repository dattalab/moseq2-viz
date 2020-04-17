from setuptools import setup, find_packages

setup(
    name='moseq2-viz',
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version='0.3.1',
    packages=find_packages(),
    platforms=['mac', 'unix'],
    install_requires=['six', 'h5py', 'tqdm', 'scipy', 'numpy', 'scipy',
                      'psutil', 'click', 'psutil', 'future', 'scikit-learn',
                      'scikit-image', 'joblib', 'seaborn', 'cytoolz', 'networkx',
                      'ipywidgets', 'matplotlib', 'statsmodels', 'hdf5storage',
                      'ruamel.yaml', 'dtaidistance', 'opencv-python'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-viz = moseq2_viz.cli:cli']}
)
