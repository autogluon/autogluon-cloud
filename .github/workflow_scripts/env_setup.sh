function setup_build_contrib_env {
    python3 -m pip install --upgrade pip
    python3 -m pip install -r $(dirname "$0")/../../docs/requirements_doc.txt
    python3 -m pip install awscli
    export AG_DOCS=1
}

function install_cloud {
    python3 -m pip install --upgrade -e .
}

function install_cloud_test {
    python3 -m pip install --upgrade pytest-xdist # Enable running tests in parallel for speedup
    python3 -m pip install --upgrade pytest-forked
    python3 -m pip install --upgrade -e ./[tests]
}
