## Environment Installation
<pre>
# Create a new conda environment with Python 3.10
conda create -n rl_env python=3.10 -y

# Activate the environment
conda activate rl_env

# Install PyTorch
pip install torch

# Install other dependencies
pip install numpy matplotlib gymnasium "gymnasium[box2d]" "gymnasium[other]"
</pre>


### Note on implementation
Timings are all reported for running the code on CPU.  GPU runs have not been tested or dubugged for this code - but you are welcome to make those changes on your own if you are interested in running on GPU.

---
HW3 Written by Matthew Bronars and Lawrence Jang