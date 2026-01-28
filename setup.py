from setuptools import setup, find_packages

setup(
    name="embodied-learning",
    version="0.1.0",
    description="A reproducible framework for embodied learning with world models",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "mujoco>=3.0.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
