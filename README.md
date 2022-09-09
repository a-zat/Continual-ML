# Unsupervised learning

## General Information
Code developed and tested on Visual Studio Code for macOS.
The dependencies can be installed running `pip install -r requirements.txt`.

### Setting the environment on VS Code
1. Open the project in VS Code
2. Create an environment for Python 3: `python3 -m venv venv`
3. Activate the environmen: `source venv/bin/activate` (for macOS). Check in the terminal that the new environmentt `(venv)` is active
4. Install the requirements: `pip install -r requirements.txt`

`.py` scripts can be executed from the terminal with `python3 main.py`, while `.ipynb` require some additional configuration:
1. Select the `venv` as Python interpreter (top right in the GUI)
2. You will be prompted to install a Jupiter kernel, click yes. You can check the installation was successful if a `venv/share/jupiter` is created
