from setuptools import setup

setup(
    name="quanteyes",
    version="0.1",
    author="TODO",
    author_email="TODO@TODO.COM",
    description="Package for quantizing gaze tracking models",
    packages=["quanteyes"],
    install_requires=["torch", "torchvision", "matplotlib", "brevitas"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)
