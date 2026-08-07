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


version = "0.5.1"
version = update_version(version)

if __name__ == "__main__":
    create_version_file(version=version)
    # Everything except the dynamic version lives in pyproject.toml.
    setup(version=version)
