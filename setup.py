import os,sys
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))

from setuptools import setup, find_packages
from lab_optimizer import __version__
    
torch_version = 'torch>=2.4.0'

setup(
    name="lab_optimizer", 
    version=__version__, 
    author="Zifeng Li",  # 
    author_email="221503020@smail.nju.edu.cn",  # 
    description="a collection of optimization algorithms",  # 
    long_description=open("README.md").read(),  # 
    long_description_content_type="text/markdown",  # 
    url="https://github.com/quantumopticss/lab_optimizer",  # 
    license='MIT',
    packages=find_packages(),  
    classifiers=[  #
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  
    install_requires=[  
        'MLOOP',
        'scikit-opt',
        'scikit-learn',
        'numpy',
        'scipy',
        'matplotlib',
        'panads',
        'seaborn',
        'plotly',
        torch_version,
    ],
)