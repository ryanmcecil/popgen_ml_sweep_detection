# StatML Sweep Detection Analyses and Tools for Population Genetics

## Articles

- *Mathematical Models Yield Insights into CNNs: Applications in Natural Image Restoration and Population Genetics* by Ryan M. Cecil.
  - Master's thesis, will be published in August 2023.
- *On convolutional neural networks for selection inference: revealing the lurking role of preprocessing, and the surprising effectiveness of summary statistics* by Ryan M. Cecil and [Lauren Sugden](https://www.duq.edu/faculty-and-staff/lauren-sugden.php).
    - [[Preprint](https://www.biorxiv.org/content/10.1101/2023.02.26.530156v1)]


## Reproduction of Analyses

The following steps and reproductions were last tested with Ubuntu 20.04. 

### Steps to get started

First, clone the repository to your computer with 

```
git clone https://github.com/ryanmcecil/popgen_ml_sweep_detection.git
```

Create a new python environment either using [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [Venv](https://docs.python.org/3/library/venv.html). Then, activate the environment and install the required python packages by calling

```
pip install -r requirements.txt
```

Afterwards, ensure that [Tensorflow](https://www.tensorflow.org/), [CUDA](https://developer.nvidia.com/cuda-toolkit), and [CudNN](https://developer.nvidia.com/cuda-toolkit) are properly installed on your computer. The code was last tested with Tensorflow v2.8.0, Cuda v11.4, and CuDNN v8.9.2. You may check that Tensorflow works and recognizes the GPU by calling

```
python -c "import tensorflow as tf;print(tf.config.list_physical_devices('GPU'))"
```
If the printed list is empty, then tensorflow does not currently recognize the GPU.

The simulation software used requires [SLiM](https://messerlab.org/slim/) to be installed on the system. Please see the manual at the link for instructions on how to install it on your system.

In addition, the `os.getcwd()` commands in the python files require that the python path contain the main root folder of this repository. You may either add the folder automatically using your code editor (for example, I use a launch.json file in my vscode editor), or add the path manually before running any code by calling

```
export PYTHONPATH="${PYTHONPATH}:$PWD"
```

Finally, to ensure that everything is in working order, please run the following command:

```
python3 test.py
```

The python code in the file will generate hard and soft sweep simulations using SLiM, convert them to images and process them using an image resizing algorithm, train a CNN on the data, then print the test accuracy.

### Running Reproduction Code

The code to reproduce the results of the above works may be found in the `reproduce` folder. The `ReadMe.md` file in the folder goes into more detail.

## Tools

- Simulation of demographic models
- Processing of simulated demographic models
- Training of CNN models
- IMplementation of common summary statistics

## Acknowledgements

If you find this repository useful for your research or applications, please cite the following [article](https://www.biorxiv.org/content/10.1101/2023.02.26.530156v1) which is currently a preprint.