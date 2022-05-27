# DiRAC Cluster Challenge 26/27 May 2022

- [Tuomas' Jupyter Notebooks](https://github.com/tkoskela/hpc_lecture_notes/tree/master/cluster_challenge/26May2022)
- [The Challenges](https://github.com/JamieJQuinn/2022-dirac-cluster-challenge)

## Getting the code

From a shell (e.g. in the terminal or on Myriad) run:

`git clone https://github.com/JamieJQuinn/2022-dirac-cluster-challenge.git`

## Python on Myriad

To get access to python 3 on Myriad, you have to load the correct module:

`module load python3`

You'll then want to create a virtual environment to store the python packages:

`python -m venv venv`

To use python inside the environment:

`source venv/bin/activate`

And now to install packages inside the environment:

`pip install -r requirements.txt`

## Running a GPU job on Myriad

Use the example script from the [Research Computing documentation](https://www.rc.ucl.ac.uk/docs/Wiki_Export/Example_Submission_Scripts/#gpu-job-script-example), name it `submit.sh` and edit accordingly.

Commands for the scheduler are preceded by `#$`, comment lines start with `#`

If you want the working directory to be the current working directory, replace `#$ -w name` with `#$ -cwd`

You can run a simple application to check the script works. a good choice is `nvidia-smi`, which just prints information about what's running on the GPU.

Submit the job script with `qsub submit.sh`, check its status with `qstat`

Now for a real GPU application! Start `vector_add.py` file.

You'll need `module load python3` before you run it. Then you have the right version of python loaded, with the needed libraries too (try `python --version` to convince yourself). Therefore, run the application with `python vector_add.py`, not `python3 vector_add.py`.

Using python environments: (especially useful if you get error messages about libraries (eg. numpy) not found)
```
python -m venv venv
source venv/bin/activate
```
Then you can `pip install` any libraries you want without polluting the system environment. eg. `pip install numpy`

## Vector add CPU

```
import numpy as np
from time import perf_counter

a = np.ones(256*4096*16)
b = np.ones(256*4096*16)

result = np.zeros_like(a)

N_ITERATIONS = 10

total_time = 0.
for i in range(N_ITERATIONS):
    start = perf_counter()
    result[:] = a[:] + b[:]
    end = perf_counter()
    total_time += end-start

print(f"Mean CPU time: {total_time/N_ITERATIONS}")
```

## Vector add GPU

```
import numpy as np
from numba import cuda
from time import perf_counter

@cuda.jit
def vec_add(out, a, b):
    i = cuda.grid(1)
    if i < a.size:
        out[i] = a[i] + b[i]

# EXAMPLE CODE, will need changing
def mat_add(out, A, B):
    i, j = cuda.grid(2)
    if i < a.shape[0] and j < a.shape[1]:
        out[i,j] = a[i,j] + b[i-1,j+1]

@cuda.jit
def do_nothing():
    i = cuda.grid(1)

print(cuda.gpus)

a = np.ones(256*4096*16, dtype=np.float64)
b = np.ones(256*4096*16, dtype=np.float64)
result = np.zeros_like(a)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_result = cuda.device_array(a.size)

N_ITERATIONS = 10

total_time = 0.
for i in range(N_ITERATIONS):
    start = perf_counter()
    result[:] = a[:] + b[:]
    end = perf_counter()
    total_time += end-start

print(f"Mean CPU time: {total_time/N_ITERATIONS}")

threadsperblock = 256
blockspergrid = (a.size + threadsperblock-1) // threadsperblock

print(threadsperblock, blockspergrid)

total_time = 0.
for i in range(N_ITERATIONS):
    start = perf_counter()
    vec_add[blockspergrid, threadsperblock](d_result, d_a, d_b)
    # do_nothing[blockspergrid, threadsperblock]()
    end = perf_counter()
    total_time += end-start

result = d_result.copy_to_host()

print(f"Mean GPU time: {total_time/N_ITERATIONS}")
```

## Myriad GPU submission script

```
#!/bin/bash -l

# Batch script to run a GPU job on Legion under SGE.

# Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=1

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:10:0

# Request 1 gigabyte of RAM (must be an integer)
#$ -l mem=5G

# Set the name of the job.
#$ -N CHANGE ME

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -cwd

export CUDA_HOME=/shared/ucl/apps/cuda/10.1.243/gnu-4.9.2

# load the cuda module (in case you are running a CUDA program
module unload compilers mpi python3
module load compilers/gnu/4.9.2
module load cuda/10.1.243/gnu-4.9.2
module load python3

# Run the application - the line below is just a random example.
source venv/bin/activate
python vector_add.py 
```

## Interactive shells on Myriad

To g
```
qrsh -l gpu=1 -now no
```

## Supported python in Numba CUDA kernels

[documentation is here](https://numba.pydata.org/numba-doc/dev/cuda/cudapysupported.html)

## Submission details

Email me your submission at [REDACTED]
