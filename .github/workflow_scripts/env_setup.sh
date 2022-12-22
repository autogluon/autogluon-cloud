function setup_lint_env {
    python3 -m pip install --upgrade pip
    python3 -m pip install tox
}

function install_cloud {
    python3 -m pip install --upgrade pytest-xdist # Enable running tests in parallel for speedup
    python3 -m pip install --upgrade -e ./[tests]
}

function install_latest_tabular_and_multimodal_dependencies {
    git clone https://github.com/autogluon/autogluon-cloud.git
    python3 -m pip install -e common/
    python3 -m pip install -e core/[all]
    python3 -m pip install -e features/
}

function install_latest_tabular {
    install_latest_tabular_and_multimodal_dependencies
    python3 -m pip install -e tabular/[all]
}

function install_latest_multimodal {
    install_latest_tabular_and_multimodal_dependencies
    python3 -m pip install -e multimodal/
}

function install_tabular {
    VERSION=$1
    if [ VERSION == "latest" ]
    then
        install_latest_tabular
    else
        python3 -m pip install -U autogluon.tabular==$1
    fi
}

function install_multimodal {
    VERSION=$1
    if [ VERSION == "latest" ]
    then
        install_latest_multimodal
    else
        python3 -m pip install -U autogluon.multimodal==$1
    fi
}
