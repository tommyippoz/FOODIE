import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='foodie-ml',
     version='0.10',
     scripts=[],
     author="Tommaso Zoppi",
     author_email="tommaso.zoppi@unifi.it",
     description="FOODIE - exercising Fail cOntrOlleD classifIErs",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/tommyippoz/FOODIE",
     keywords=['fail-controlled classifier', 'fail-omission', 'machine learning', 'safety wrapper', 'safety monitor',
               'uncertainty measures', 'ensemble'],
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )