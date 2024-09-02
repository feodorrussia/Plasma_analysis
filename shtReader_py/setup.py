import setuptools
import codecs

with codecs.open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="shtRipper_cpp",
    version="1.3.3",
    author="Rezenter",
    author_email="nisovru@gmail.com",
    description="C++ parser of .sht files.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rezenter/shtRipper_cpp/tree/master/python",
    packages=['shtRipper'],
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        'Operating System :: Microsoft :: Windows :: Windows 10',
    ],
    python_requires='>=3.5',
    install_requires=[],
    entry_points={
        'console_scripts': [
            'cursive = cursive.tools.cmd:cursive_command',
        ],
    },
    include_package_data=True
)
