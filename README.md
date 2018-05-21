# jpsi-ml-classifier
Machine-learning-based system mimicking the student's decision in Particle Analysis lab activity.

**This repository is a work in progress. Please come back for more changes over time.**

## Installation
Python 3 and a handful of packages are required.

### TL;DR (for experienced users)
1. Install Python 3 and the following packages on your system:
    * jupyter
    * scikit-learn
    * pandas
    * seaborn
    * matplotlib
    * numpy
2. Clone this repository using Git into the desired location.

### On Linux
We recommend running this program with Ubuntu, as it is the most popular Linux distribution. Other Linux distributions will also work, but installation steps may differ.
1. Open the Terminal by finding it from GUI (depending on your Linux GUI configuration), or press `Ctrl + Alt + T`.
2. Once you are in the Terminal, make sure you have installed Python 3 first. Different Linux distributions (distros) have different package managers, so for Ubuntu, type:
    ```bash
    sudo apt-get install python3 python3-dev
    ```
    `python3` is the standard package for Python 3, while `python3-dev` is the additional package where future Python packages can use to compile and install onto your system. You will be required to enter the root password. If Python 3 is not installed on your system, this command will install it for you. If you are using a Linux distro other than Ubuntu, you will need a different package manager other than `apt`, and . Consult with your Linux distro's documentation for more info.
3. Install required Python packages. Chances are your Python 3 installation already includes Python package manager `pip`, so, in the same Terminal window, type:
    ```bash
    sudo pip install jupyter scikit-learn pandas seaborn matplotlib numpy
    ```
    `pip` will install these three packages and all other packages required by these three packages if they are not already on your computer. If `pip` is not available on your system, simply install `python3-pip` package from `apt-get` with:
    ```bash
    sudo apt-get install python3-pip
    ```
4. Clone this repository with Git. Navigate to the desired folder, and type:
    ```bash
    git clone https://github.com/vicha-w/jpsi-ml-classifier
    ```
    If your system does not have Git installed, you can install it using the package manager from your Linux distro.

### On Windows
We do not recommend installing Python 3 using official installer from python.org, since it can mess up your system and is hard to uninstall. Instead, we recommend installing Anaconda with Python 3. Anaconda is a Python distribution, complete with package manager and popular packages used in data science, all in a nice and tidy installation.

1. Download Anaconda distribution **with Python 3** from www.anaconda.com/download/
2. Run the installer and follow the instructions.
3. Clone this repository with Git, or download the whole repository as a zip file from this page. 

If you are running Windows 10, you may enable Windows Subsystem for Linux (WSL), which enables your Windows PC to install your preferred Linux distro as a subsystem, complete with integration to your files on Windows. Once you have completed the installation, you may follow the same installation instructions [on Linux](#on-linux). For more information, go to docs.microsoft.com/en-us/windows/wsl/install-win10.

## Usage
1. **Download the dataset from CERN OpenData portal** The dataset has 20 parts, each containing 100 collision events. On linux, we have prepared bash script for this.
2. **Download the accompanying spreadsheet [here](http://opendata.cern.ch/record/301/files/dimuon-Jpsi.csv)** This spreadsheet contains information on muon energy and momentum.
3. **Run prepareAllText.py**

