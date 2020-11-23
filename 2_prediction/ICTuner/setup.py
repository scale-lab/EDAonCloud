from setuptools import setup, find_packages

# version
version = {}
with open('ictuner/version.py', 'r') as f:
    exec(f.read(), version)
download_url = 'https://github.com/EDAAC/ICTuner/archive/v_' + version['__version__'] + '.tar.gz'

# description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'ictuner',
  packages = find_packages(),
  version = version['__version__'],
  license = 'BSD-3 License',
  description = 'Auto-tuning physical design flows',
  author = 'Abdelrahman Hosny',
  author_email = 'abdelrahman_hosny@brown.edu',
  url = 'https://github.com/EDAAC/ICTuner',
  download_url = download_url,
  keywords = ['EDA', 'Electronic', 'Design', 'Automation', 'Parameter', 'Tuning', 'Search', 'Exploration'],
  install_requires=[
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
  long_description = long_description,
  long_description_content_type = 'text/markdown',
)