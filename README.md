# CNN-ripple detector for MATLAB

This ripple detector uses a Convolutional Neural Network (CNN) originally created for python in [this repository](https://github.com/PridaLab/cnn-ripple). This code is a version that internally calls the python function to output a ripple probability along time.

_**UPDATE!** We have added the option to discard False Positives that come from extreme offset drifts on the recording signal. Just as muscular artifacts affect filter detection of SWRs, extreme offset noise can affect the CNN, but it has been already implemented a solution to counteract this noise._ 

_**UPDATE!** Also, the CNN predicts now much faster when the sliding window (called `pred_every`) is less than 32ms._ 

## Installation

For this script you should have [anaconda](https://www.anaconda.com/products/distribution) installed. Here are the steps to install python packages.

1. Open your anaconda terminal, and go to your "cnn-matlab" folder by typing:

		cd <full path to cnn_matlab folder>

2. Then make a new anaconda environment with:

		conda create --name cnn-env python=3.8 tensorflow=2.3 keras=2.4 numpy=1.18 h5py=2.10

3. Copy the path of the environment that outputs the following command:

		conda activate cnn-env
		echo %CONDA_PREFIX%

4. That path corresponds to the optional input variable "exec_env", an input of `detect_ripples_cnn()` function.


## Using Tensorflow from Matlab

Every time you want to use the CNN for SWR detection, you will have to open the anaconda terminal, and execute MATLAB from there, typing:
	
	matlab


## Detecting ripples

### detect_ripples_cnn()

The MATLAB function `detect_ripples_cnn(data, fs, <optionals>)` calls python to compute the sharp-wave ripple probability along time. It receives:

* **Mandatory inputs**
	- `data`: `n_samples` x `n_channels`. LFP data (in double format). As the CNN works with 8 channels, by default it will take the first 8 channels.
	- `fs`: sampling frequency

* **Optional inputs**
	- `exec_env`: Full path to python executable of the `cnn-env` containing tensorflow (output of installation step 3)
	- `channels`: List of 8 channels to use for the prediction. By default it takes the first 8 channels
	- `model_file`: Full path to folder where the CNN model is stored. By default searches in 'cnn/' inside the folder containing this file.
	- `pred_every`: (optional) Prediction window size. By default is 0.032 seconds, but it can be change to any other value less than that.

* **Output**
	- `n_samples` x 1 array of the probability (between 0 and 1) of a sample to be a sharp-wave ripple.


### get_intervals()

In order to get time intervals of SWRs, a threshold must be chosen. You can use `get_intervals(SWR_prob, <optionals>)`, that has usage modes depending on the inputs:

* **Setting a threshold**
	- `get_intervals(SWR_prob)`: displays a histogram of all SWR probability values, and a draggable threshold to set a threshold based on the values of this particular session. When 'Done' button is pressed, the GUI takes the value of the draggable as the threshold.
	- `get_intervals(SWR_prob, 'LFP', data, 'fs', fs, 'win_size', win_size)`: if `data` is also added as an input, then the GUI adds up to 50 examples of SWR detections. If the 'Update' button is pressed, another 50 random detections are shown. When 'Done' button is pressed, the GUI takes the value of the draggable as the threshold. Sampling frequency `fs` (in Hz) and window size `win_size` (in seconds) can be used to set the window length of the displayed examples. It automatically discards false positives due to drifts, but if you want to set it off, you can set `discard_drift` to `false`. By default, it discards noises whose mean LFP is above `std_discard` times the standard deviation, which by default is 1SD. This parameter can also be changed.
	- `get_intervals(SWR_prob, 'threshold', threshold)`: if a threshold is given, then it takes that threshold without displaying any GUI.

* **Getting the intervals**
	Once the threshold is set, it automatically finds the intervals where `SWR_prob` is over that threshold. The output of `get_intervals()` is a `n_ripples` x 2 array of beginnings and ends. If `fs` has been given, the output is in seconds; if it's not given, then the output is in timestamps.


## Example

In the [test folder](https://github.com/PridaLab/cnn-matlab/tree/master/test) you can find a test main script `test_cnn_matlab.m` that downloads a recording from figshare and runs the CNN detection for several configurations, including dead channels.
