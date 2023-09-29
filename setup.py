import os

from setuptools import setup

AUTOGLUON = "autogluon"
CLOUD = "cloud"

PYTHON_REQUIRES = ">=3.8, <3.11"


def create_version_file(*, version):
    print("-- Building version " + version)
    version_path = os.path.join("src", AUTOGLUON, CLOUD, "version.py")
    with open(version_path, "w") as f:
        f.write(f'"""This is the {AUTOGLUON}.{CLOUD} version file."""\n')
        f.write("__version__ = '{}'\n".format(version))


def update_version(version, use_file_if_exists=True, create_file=False):
    """
    To release a new stable version on PyPi, simply tag the release on github, and the Github CI will automatically publish
    a new stable version to PyPi using the configurations in .github/workflows/pypi_release.yml .
    You need to increase the version number after stable release, so that the nightly pypi can work properly.
    """
    try:
        if not os.getenv("RELEASE"):
            from datetime import date

            minor_version_file_path = "VERSION.minor"
            if use_file_if_exists and os.path.isfile(minor_version_file_path):
                with open(minor_version_file_path) as f:
                    day = f.read().strip()
            else:
                today = date.today()
                day = today.strftime("b%Y%m%d")
            version += day
    except Exception:
        pass
    if create_file and not os.getenv("RELEASE"):
        with open("VERSION.minor", "w") as f:
            f.write(day)
    return version


def default_setup_args(*, version):
    from setuptools import find_packages

    long_description = open("README.md").read()
    name = f"{AUTOGLUON}.{CLOUD}"
    setup_args = dict(
        name=name,
        version=version,
        author="AutoGluon Community",
        url="https://github.com/autogluon/autogluon-cloud",
        description="Train and deploy AutoGluon backed models on the cloud",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="Apache-2.0",
        license_files=("LICENSE", "NOTICE"),
        # Package info
        packages=find_packages("src"),
        package_dir={"": "src"},
        namespace_packages=[AUTOGLUON],
        zip_safe=True,
        include_package_data=True,
        python_requires=PYTHON_REQUIRES,
        package_data={
            AUTOGLUON: [
                "LICENSE",
            ]
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Education",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Customer Service",
            "Intended Audience :: Financial and Insurance Industry",
            "Intended Audience :: Healthcare Industry",
            "Intended Audience :: Telecommunications Industry",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Image Recognition",
        ],
        project_urls={
            "Documentation": "https://auto.gluon.ai",
            "Bug Reports": "https://github.com/autogluon/autogluon-cloud/issues",
            "Source": "https://github.com/autogluon/autogluon-cloud/",
            "Contribute!": "https://github.com/autogluon/autogluon-cloud/blob/master/CONTRIBUTING.md",
        },
    )
    return setup_args


version = "0.2.1"
version = update_version(version, use_file_if_exists=False, create_file=True)

install_requires = [
    # common module provides utils with stable api across minor version
    "autogluon.common>=0.7,<1.0",
    # <2 because unlikely to introduce breaking changes in minor releases. >=1.10 because 1.10 is 3 years old, no need to support older
    "boto3>=1.10,<2.0",
    "numpy>=1.21,<1.27",
    "packaging>=23.0,<24.0",
    "pandas>=1.4.1,<1.6",
    # updated sagemaker is required to fetch latest container info, so we don't want to cap the version too strict
    # otherwise cloud module needs to be released to support new container
    "sagemaker>=2.126.0,<3.0",
    "pyarrow>=11.0,<11.1",
    "PyYAML~=6.0",
    "Pillow>=9.3.0,<10.0",  # unlikely to introduce breaking changes in minor releases
    "ray[default]>=2.3.0,<2.4.0",
]

extras_require = dict()

all_requires = ["autogluon>=0.7,<1.0"]  # To allow user to pass ag objects
extras_require["all"] = all_requires

test_requirements = [
    "tox",
    "pytest",
    "pytest-cov",
    "moto[all]",
    "autogluon.common>=0.7.0b,<1.0",
]  # Install pre-release of common for testing

test_requirements = list(set(test_requirements))
extras_require["tests"] = test_requirements

if __name__ == "__main__":
    create_version_file(version=version)
    setup_args = default_setup_args(version=version)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
