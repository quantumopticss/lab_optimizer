from setuptools import setup, find_packages

setup(
    name="lab_optimizer", 
    version="1.0.2", 
    author="Zifeng Li",  # 
    author_email="221503020@smail.nju.edu.cn",  # 
    description="a collection of optimization algorithms",  # 
    long_description=open("README.md").read(),  # 
    long_description_content_type="text/markdown",  # 
    url="https://github.com/quantumopticss/lab_optimizer",  # 
    packages=find_packages(),  
    classifiers=[  #
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  
    install_requires=[  
        "requests",  
    ],
)