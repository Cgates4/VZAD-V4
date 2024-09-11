## Work in Progress
# Work in Progress
Recommended IDE: PyCharm newest version
Install depenencies needed to run code:

## Install GNU
## Install NLOPT (if nlopt can't be downloaded from respective IDE)
Install Swig (dependency for nlopt) by going to swig hompeage (https://www.swig.org/download.html) or using homebrew
Install homebrew by following directions from https://brew.sh/ 
brew install swig

Intall numpy (dependency needed for nlopt), numpy homepage: https://numpy.org/

Download nlopt zip file, nlopt homepage: https://nlopt.readthedocs.io/en/latest/
After downloading and unzipping the nlopt zip file, from the command line move into the nlopt folder and enter the following promts:
cmake -DNLOPT_GUILE=OFF -DNLOPT_MATLAB=OFF -DNLOPT_OCTAVE=OFF -DNLOPT_TESTS=OFF -DPYTHON_EXECUTABLE=/usr/local/bin/python3 or cmake -DPython_EXECUTABLE=/usr/bin/python3.12
make
sudo make install

See if python bindings can be imported
python3 -c "import nlopt; print(nlopt.__version__)"


https://stackoverflow.com/questions/62704802/cannot-install-nlopt-python-module

# DONT FOLLOW STEPS BELOW (These are for c++ vzad)
Install GNU
File: https://www.gnu.org/software/gsl/
Instructions: https://gist.github.com/TysonRayJones/af7bedcdb8dc59868c7966232b4da903

## Install PyGSL
Install PyGSL
pygsl tar file: https://github.com/pygsl/pygsl/releases 
pip install pygsl or https://pypi.org/project/pygsl/
If want to set up manually (issues may occur):
cd desktop (or where your file is)
gzip -d -c pygsl-2.3.3.tar.gz | tar xvf -
cd pygsl-2.3.3
python setup.py gsl_wrappers
python setup.py config
sudo python setup.py install

## Install NLOPT
After downloading and unzipping the nlopt zip file, from the command line move into the nlopt folder and enter the following promts:
mkdir build
cd build

Move the nlopt.py file into site-packages if it is not already there.

For help: https://stackoverflow.com/questions/62704802/cannot-install-nlopt-python-module# VZAD-V4
VZAD Version 4 (Python)
