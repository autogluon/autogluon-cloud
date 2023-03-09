function setup_lint_env {
    python3 -m pip install --upgrade pip
    python3 -m pip install tox
}

function setup_build_contrib_env {
    python3 -m pip install --upgrade pip
    python3 -m pip install -r $(dirname "$0")/../../docs/requirements_doc.txt
    python3 -m pip install git+https://github.com/zhanghang1989/d2l-book
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

function install_latest_tabular_and_multimodal_dependencies {
    git clone https://github.com/autogluon/autogluon.git
    python3 -m pip install -e autogluon/common/
    python3 -m pip install -e autogluon/core/[all]
    python3 -m pip install -e autogluon/features/
}

function install_latest_tabular {
    install_latest_tabular_and_multimodal_dependencies
    python3 -m pip install -e autogluon/tabular/[all]
}

function install_latest_multimodal {
    install_latest_tabular_and_multimodal_dependencies
    python3 -m pip install -e autogluon/multimodal/
}

function install_tabular {
    VERSION=$1
    if [ $VERSION = "source" ]
    then
        install_latest_tabular
    else
        python3 -m pip install -U autogluon.tabular==$1
    fi
}

function install_multimodal {
    VERSION=$1
    if [ $VERSION = "source" ]
    then
        install_latest_multimodal
    else
        python3 -m pip install -U autogluon.multimodal==$1
    fi
}
