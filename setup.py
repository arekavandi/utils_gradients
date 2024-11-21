from distutils.core import setup

setup(
    name='utils_gradients',
    version='0.1dev',
    packages=['utils_gradients',],
    license='LICENSE.md',
    description='Useful Functions in fMRI Data Analysis',
    long_description=open('README.md').read(),
    author = ['Aref Miri Rekavandi'],
    install_requires=[
        "numpy","scipy","tqdm","matplotlib"
    ],
)
