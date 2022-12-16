function setup_build_env {
    python3 -m pip install --upgrade pip
    python3 -m pip install tox
    python3 -m pip install flake8
    python3 -m pip install black>=22.3
    python3 -m pip install isort>=5.10
}

function install_cloud {
    python3 -m pip install --upgrade pytest-xdist # Enable running tests in parallel for speedup
    python3 -m pip install --upgrade -e ./[tests]
}
