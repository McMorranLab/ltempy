from setuptools import setup, find_packages

setup(
	name = 'ltempy',
	packages = find_packages(),
	version = '1.3.0',
	author = 'William S. Parker',
	author_email = 'will.parker0@gmail.com',
	description = 'Utilities for Lorentz TEM data analysis and simulation.',
	url = 'https://github.com/McMorranLab/ltempy',
	project_urls={
		"Documentation" : "https://mcmorranlab.github.io/ltempy/",
		"Bug Tracker": "https://github.com/McMorranLab/ltempy/issues",
	},
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Operating System :: OS Independent",
	],
	long_description = open('README.md').read(),
	long_description_content_type = "text/markdown",
	python_requires='>=3.6',
	install_requires=['numpy', 'matplotlib']
)
