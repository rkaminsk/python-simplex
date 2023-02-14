from setuptools import setup

setup(
    version="1.0.0",
    name="simplex",
    description="An educational simplex solver.",
    long_description_content_type="text/markdown",
    author="Roland Kaminski",
    author_email="kaminski@cs.uni-potsdam.de",
    license="MIT",
    url="https://github.com/rkaminsk/python-simplex",
    packages=["simplex"],
    package_data={"simplex": ["py.typed"]},
    zip_safe=False,
    python_requires=">=3.7",
)

"""
This is provided for compatibility.
"""
from setuptools import setup

setup()

