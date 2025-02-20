from setuptools import setup, find_packages

setup(
    name="lab_optimizer", 
    version="1.3.1", 
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
        'M-LOOP>=3.3.5',
        'scikit-opt>=0.6.6',
        'scikit-learn>=1.2.2',
        'numpy>=1.26.4',
        'scipy>=1.14.0',
        'matplotlib>=3.9.2',
        'pandas>=1.5.3',
        'seaborn>=0.12.2',
        'plotly>=5.24.1',
        'torch>=2.3.1',
        # 'gpytorch>=1.14'
    ],
)