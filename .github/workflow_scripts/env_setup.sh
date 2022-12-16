function setup_lint_env {
    python3 -m pip install --upgrade pip
    python3 -m pip install tox
}

function install_cloud {
    python3 -m pip install --upgrade pytest-xdist # Enable running tests in parallel for speedup
    python3 -m pip install --upgrade -e ./[tests]
}
