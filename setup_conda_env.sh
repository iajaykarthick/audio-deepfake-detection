#!/bin/bash
 
# Set variables
ENV_NAME="audio_deepfake_env"
PYTHON_VERSION="3.9" 
REQUIREMENTS_FILE="requirements.txt"
PYTHONPATH_DIR="$(pwd)/src"


# Check if the conda environment already exists
if conda info --envs | grep -q "^$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Removing the existing environment."
    conda remove --name $ENV_NAME --all -y
fi

# Create the conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate the environment
source activate $ENV_NAME

# Create the activate.d directory
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

# Create the deactivate.d directory
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# Create the activation script
cat <<EOL > $CONDA_PREFIX/etc/conda/activate.d/set_pythonpath.sh
#!/bin/sh
export OLD_PYTHONPATH=\$PYTHONPATH
export PYTHONPATH=$PYTHONPATH_DIR:\$PYTHONPATH
EOL

# Create the deactivation script
cat <<EOL > $CONDA_PREFIX/etc/conda/deactivate.d/unset_pythonpath.sh
#!/bin/sh
export PYTHONPATH=\$OLD_PYTHONPATH
unset OLD_PYTHONPATH
EOL

# Make the scripts executable
chmod +x $CONDA_PREFIX/etc/conda/activate.d/set_pythonpath.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/unset_pythonpath.sh

# Install dependencies from requirements.txt
pip install -r $REQUIREMENTS_FILE

# Verify the setup
echo "PYTHONPATH is set to: \$PYTHONPATH"
echo "Environment setup complete. To activate the environment, run: conda activate $ENV_NAME"
