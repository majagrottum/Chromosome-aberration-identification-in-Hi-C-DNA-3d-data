from setuptools import setup

setup(
    name = 'cluster_visualization',
    version = '0.1.0',
    packages = ['cluster_visualization'],
    install_requires=[ 'numpy', 'networkx', 'node2vec', 'hdbscan', 'scikit-learn', 'matplotlib',],
    entry_points = {
        'console_scripts': [
            'cluster_visualization = cluster_visualization.__main__:main',
        ]
    })
