conda activate f25-703-HW3
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
python3 runner.py --agent ppo