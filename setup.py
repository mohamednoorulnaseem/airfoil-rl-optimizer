from setuptools import setup, find_packages

setup(
    name='airfoil-rl-optimizer',
    version='1.0.0',
    description='RL-based airfoil optimization with XFOIL CFD validation',
    author='Mohamed Noorul Naseem',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'stable-baselines3>=2.0.0',
        'gymnasium>=0.28.0',
        'dash>=2.14.0',
        'plotly>=5.17.0',
    ],
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)
