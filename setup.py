from setuptools import setup, find_packages

setup(
    name='capomate',
    version='0.1alpha',
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    license='MIT License',
    long_description=open('README.md').read(),
)
