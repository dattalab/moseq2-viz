from setuptools import setup, find_packages

setup(
    name='moseq2-viz',
    author='Jeff Markowitz',
    description='To boldly go where no mouse has gone before',
    version='0.3.0',
    packages=find_packages(),
    platforms=['mac', 'unix'],
    install_requires=[],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['moseq2-viz = moseq2_viz.cli:cli']}
)
