jupyter lab build
pip uninstall jupyterlab -y
pip install jupyterlab
jupyter lab clean
jupyter lab build
jupyter labextension list
pip install --upgrade jupyter jupyterlab ipykernel
