import numpy as np
import sys
import os

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(THIS_PATH, 'cnn'))
import model_class

# Globals
downsampled_fs = 1250.
window_size = 0.032
stride_eff = 0.032
num_filters_factor = 1.

def resample_data (data, from_fs, to_fs):

    # Dowsampling
    resampled_pts = np.linspace(0, data.shape[0]-1, int(np.round(data.shape[0]/from_fs*to_fs))).astype(int)
    resampled_data = data[resampled_pts, :]

    # Change from int16 to float16 if necessary
    # int16 ranges from -32,768 to 32,767
    # float16 has ±65,504, with precision up to 0.0000000596046
    if resampled_data.dtype != 'float16':
        resampled_data = np.array(resampled_data, dtype="float16")

    return resampled_data


def z_score_normalization(data):
	channels = range(np.shape(data)[1])

	for channel in channels:
		# Since data is in float16 type, we make it smaller to avoid overflows
		# and then we restore it.
		# Mean and std use float64 to have enough space
		# Then we convert the data back to float16
		dmax = np.amax(data[:, channel])
		dmin = abs(np.amin(data[:, channel]))
		dabs = dmax if dmax>dmin else dmin
		m = np.mean(data[:, channel] / dmax, dtype='float64') * dmax
		s = np.std(data[:, channel] / dmax, dtype='float64') * dmax
		s = 1 if s == 0 else s # If std == 0, change it to 1, so data-mean = 0
		data[:, channel] = ((data[:, channel] - m) / s).astype('float16')
	
	return data


def generate_overlapping_windows_fast(data, stride, fs):
	'''
	Expand data by concatenating windows according to window_size and stride

	Inputs:
	-------
		data: numpy array (n_samples x n_channels) 
			LFP data

		stride: float (s)
			Length of stride in seconds (step taken by the window). Note that window size is given
			by the model (currently 32ms)

		fs: integer (Hz)
			sampling frequency in Hz of LFP data 


	Outputs: 
	--------
		new_data: numpy array (1, n_samples', n_channels)
			Numpy array containing the expanded data.

	23-May-2022: Julio E
	'''

	assert window_size>=stride, 'stride must be smaller or equal than window size (32ms) to avoid discontinuities'
	window_pts = int(window_size * fs)
	stride_pts = int(stride * fs)
	assert stride_pts>0, 'pred_every must be larger or equal than 1/downsampled_fs (>0.8 ms)'
	num_windows = np.ceil((data.shape[0]-window_pts)/stride_pts).astype(int)+1 
	remaining_pts = (num_windows-1)*stride_pts + window_pts - data.shape[0]
	new_data = np.zeros(((num_windows+1)*window_pts,data.shape[1])) #add one empty window for the cnn

	for win_idx in range(num_windows-1):
		win = data[win_idx*stride_pts:win_idx*stride_pts+window_pts,:]
		new_data[win_idx*window_pts:(win_idx+1)*window_pts,:]  = win

	new_data[(win_idx+1)*window_pts:-remaining_pts-window_pts,:] = data[(win_idx+1)*stride_pts:, :]
	new_data = np.expand_dims(new_data, 0)

	return new_data


def integrate_window_to_sample(win_data, stride, fs, n_samples=None, func=np.mean):
	'''
	Expand data from windows to original samples taking into account stride size

	Inputs:
	-------
		win_data: numpy array (n_windows,) 
			data for each window to be expanded into samples

		stride: float (s)
			Length of stride in seconds (step taken by the window). Note that window size is given
			by the model (currently 32ms)

		fs: integer (Hz)
			sampling frequency in Hz

		n_samples: integer
			desired number of samples. For instance, last window may be half empty (due to zero paddings).

		func: arithmetic function
			function to be applied when there is more than one window referencing the same sample (
			overlapping due to stride/window_size missmatch).

	Outputs: 
	--------
		new_data: numpy array (1, n_samples', n_channels)
			Numpy array containing the expanded data.

	23-May-2022: Julio E
	'''

	assert window_size>=stride, 'stride must be smaller or equal than window size (32ms) to avoid discontinuities'
	window_pts = int(window_size * fs)
	stride_pts = int(stride * fs)
	assert stride_pts>0, 'pred_every must be larger or equal than 1/downsampled_fs (>0.8 ms)'

	max_win_overlap = np.ceil(window_pts/stride_pts).astype(int) 
	max_num_win = win_data.shape[0]

	if isinstance(n_samples, type(None)):
		n_samples = (max_num_win-1)*stride_pts + window_pts

	sample_data = np.empty((n_samples,))
	win_list = []
	for sample in range(0, n_samples, stride_pts):
		if len(win_list) == 0: #first stride simply append window 0
			win_list.append(0)
		else:
			win_list.append(win_list[-1]+1) #append new window
			if len(win_list)>max_win_overlap: #pop left-most window if aready maximum overlapping
			    win_list.pop(0)
			if win_list[-1]>=max_num_win: #discard added window if beyond maximum number of windows
			    win_list.pop(-1)
		sample_data[sample:sample+stride_pts] = func(win_data[win_list])

	return sample_data


def predict(data_original, channels_list, fs, model_file, pred_every=window_size, handle_overlap = 'mean', verbose=False):
	'''
	This function outputs a SWR probability along time:
		
		predict(data, channels_list, fs, model_file, threshold, pred_every=window_size, verbose=False)

	Inputs:

		data:			LFP data of size (n_samples x n_channels)
		channels_list:	Array of 8 channels over which make SWR predictions
		fs: 			Sampling frequency in Hz
		model_file:		Full folder path in which CNN model is stored
		pred_every: 	(optional) Prediction window size. By default is 0.032 seconds,
						but it can be change to any other value less than that, at expense
						of taking a considerably more amount of time
		handle_overlap  (optional) Determines the way 
		verbose:		(optional) Print messages. By default is False
	
	Outputs:

		prediction: 	SWR probability along time at original fs sampling frequency

	'''

	# Transform to numpy
	channels_list = np.array(channels_list).astype('int') - 1 # Indexes in MATLAB convention, so we substract 1
	data = np.array(data_original)

	# Errors
	if len(channels_list) != 8:
		raise Exception('Input "channel_list" must contain 8 elements')
	if any(channels_list<0) or any(channels_list>=data.shape[1]):
		raise Exception('Input "channel_list" contains invalid channel numbers')
	assert isinstance(handle_overlap, str), "input 'handle_overlap' must be a string ('mean' or 'max')"
	if 'mean' in handle_overlap:
		f_overlap = np.mean
	elif 'max' in handle_overlap:
		f_overlap = np.max
	else:
		raise Exception("Input "handle_overlap" must contain be 'mean' or 'max'")

	if verbose:
		print("Input data shape: ", data.shape)
		print("Channels list: ", channels_list)

	# Transform data to needed format: 8 channels, downsampled and z-scored
	data = data[:, channels_list]
	if verbose:
		print(f"\nDownsampling data from {fs} Hz to {downsampled_fs} Hz...", end=" ")
	data = resample_data(data, from_fs=fs, to_fs=downsampled_fs)
	if verbose:
		print("Done")

	if verbose:	
		print("z-scoring channels...", end=" ")
	data = z_score_normalization(data)
	if verbose:
		print("Done")
	if verbose:
		print("Resulting data shape: ", data.shape)


	# Load model
	if verbose: 
		print(f"\nLoading CNN model from folder {model_file}:")
	model = model_class.ModelClass(downsampled_fs)
	kernel_size, stride, padding, layer_type, num_filters = model.set_architecture(window_size, stride_eff, num_filters_factor)
	model.load_model(model_file)
	if verbose:
		model.model_summary()


	# Predict
	if verbose:
		print("Generating windows...", end=" ")
	lfp_cnn = generate_overlapping_windows_fast(data, pred_every, downsampled_fs)
	if verbose:
		print("Done")
		print("Resulting data shape: ", data.shape)


	if verbose:
		print("Detecting ripples:")
	y_pred_win = model.model_predict(lfp_cnn).flatten()


	# One prediction per sample (instead of per window)
	y_pred_sample = integrate_window_to_sample(y_pred_win, pred_every, downsampled_fs, n_samples=data.shape[0], func=f_overlap)

	# Transform to original fs
	prediction_fs = resample_data(y_pred_sample.reshape(-1,1), from_fs=downsampled_fs, to_fs=fs).flatten()
	if verbose:
		print("Done")

	return prediction_fs


