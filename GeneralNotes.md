# Notes, hints, links



## Python environments

- [Good doc link](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
- Key commands:
  -  `python3 -m venv .venv` - Create environment in the current folder 
  -  `source .venv/bin/activate` - Activate an environment (run from root of project)
  - `deactivate` - Deactivate an environment
  - `pip freeze > requirements.txt` - extract venv's pip packages into a requirements file

## Visual Studio Code

- Debug code under a virtual environment - https://stackoverflow.com/questions/54009081/how-can-i-debug-a-python-code-in-a-virtual-environment-using-vscode

  "Make sure the environment you want to use is selected in the Python extension for VS Code by running the **Select Interpreter** command or via the status bar. Otherwise you can explicitly set the Python interpreter to be used when debugging via the python setting for your debug config."


