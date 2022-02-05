"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

from distutils.core import setup
from version import version

setup(name='chumpy',
    version=version,
    #py_modules=['ch', 'ch_ops', 'linalg', 'utils', 'api_compatibility', 'ch_random', 'test_ch', 'test_inner_composition', 'test_linalg'],
    packages = ['chumpy'],
    package_dir = {'chumpy': '.'},
    author='Matthew Loper',
    author_email='matt.loper@gmail.com',
    url='https://github.com/mattloper/chumpy',
    description='chumpy',
    license='MIT',
    install_requires=['numpy >= 1.8.1', 'scipy >= 0.13.0'],

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux'        
    ],
)

