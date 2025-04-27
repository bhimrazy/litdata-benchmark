# Install ffcv dependencies
conda install -y -c conda-forge libjpeg-turbo
conda install -y pkg-config compilers opencv=4.6 -c conda-forge
pip install --force-reinstall 'numpy<2' 'numpy>=1.21'
pip uninstall -y opencv-python-headless numba
conda install -y opencv=4.6
pip install opencv-python-headless numba
pip install ffcv