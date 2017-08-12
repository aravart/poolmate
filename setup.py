from setuptools import setup

setup(
    name='poolmate',
    version='0.1a0',
    packages=['poolmate'],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    license='MIT License',
    install_requires=[
        'pandas',
        'numpy',
        'tqdm',
        'scipy',
        'sklearn',
    ],
    long_description=open('README.md').read(),
)
