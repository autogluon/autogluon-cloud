import os

from setuptools import setup

AUTOGLUON = "autogluon"
CLOUD = "cloud"


def create_version_file(*, version):
    print("-- Building version " + version)
    version_path = os.path.join("src", AUTOGLUON, CLOUD, "version.py")
    with open(version_path, "w") as f:
        f.write(f'"""This is the {AUTOGLUON}.{CLOUD} version file."""\n')
        f.write("__version__ = '{}'\n".format(version))


def update_version(version):
    """
    To release a new stable version on PyPi, tag the release on github; the Github CI publishes it to PyPi
    (see .github/workflows/pypi_release.yml, which sets RELEASE=1). Non-release builds get a `.dev0` suffix.
    """
    if not os.getenv("RELEASE"):
        version += ".dev0"
    return version


def default_setup_args(*, version):
    from setuptools import find_namespace_packages

    setup_args = dict(
        version=version,
        packages=find_namespace_packages("src"),
        package_dir={"": "src"},
        zip_safe=True,
        include_package_data=True,
        package_data={
            "autogluon.cloud": [
                "default_cluster_configs/*.yaml",
                "utils/autogluon_dlc.json",
                "templates/*.yaml",
            ],
        },
    )
    return setup_args


version = "0.5.1"
version = update_version(version)

install_requires = [
    # common module provides utils with stable api across minor version
    "autogluon.common>=0.7,<1.6",
    # <2 because unlikely to introduce breaking changes in minor releases. >=1.10 because 1.10 is 3 years old, no need to support older
    "boto3>=1.10,<2",
    "packaging>=23.0,<27",
    "sagemaker>=2.126.0,<3",
    "pyarrow>=19.0.1,<25",  # lower bound to avoid https://github.com/apache/arrow/issues/45283
    "PyYAML~=6.0",
    "Pillow>=10.2,<13",
    "huggingface_hub>=0.20,<2",
    "typing_extensions>=4.0,<5",
    # CLI dependencies (autogluon-cloud command)
    "click>=8.0,<9",
    "rich>=13.0,<15",
]

extras_require = dict()

ray_requires = ["ray[default]>=2.10.0,<2.56"]
extras_require["ray"] = ray_requires

all_requires = ["autogluon>=0.7,<1.6"] + ray_requires  # To allow user to pass ag objects
extras_require["all"] = all_requires

test_requirements = [
    "pytest",
    "moto",
    "autogluon.common>=0.7",
]
extras_require["tests"] = test_requirements

if __name__ == "__main__":
    create_version_file(version=version)
    setup_args = default_setup_args(version=version)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
