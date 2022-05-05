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


def predict(data_original, channels_list, fs, model_file, pred_every=window_size, verbose=False):
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
	if any(channels_list<=0) or any(channels_list>=data.shape[1]):
		raise Exception('Input "channel_list" contains invalid channel numbers')

	if verbose:
		print("Input data shape: ", data.shape)
		print("Channels list: ", channels_list)


	# Transform data to needed format: 8 channels, downsampled and z-scored
	data = data[:, channels_list]
	data = resample_data(data, from_fs=fs, to_fs=downsampled_fs)
	data = z_score_normalization(data)
	data = np.expand_dims(data, axis=0)
	if verbose:
		print("Resulting data shape: ", data.shape)


	# Load model
	model = model_class.ModelClass(downsampled_fs)
	kernel_size, stride, padding, layer_type, num_filters = model.set_architecture(window_size, stride_eff, num_filters_factor)
	model.load_model(model_file)
	model.model_summary()

	
	# Predict
	every = int(pred_every * downsampled_fs)
	if pred_every == 0.032:
		y_pred = model.model_predict(data, batch_size=1, verbose=True)
	else:
		lfpred = np.array([lfpred[i:i+window_len] for i in range(0,len(idxs)-window_len-1,every)])
		y_pred = model.model_predict(lfpred).reshape(1,-1)

	# One prediction per sample (instead of per window)
	prediction_extended = np.tile(y_pred.flatten(), (every,1)).T.flatten()

	# Transform to original fs
	prediction_fs = resample_data(prediction_extended.reshape(-1,1), from_fs=downsampled_fs, to_fs=fs).flatten()
	prediction_fs = np.append(prediction_fs, np.zeros(data_original.shape[0]-len(prediction_fs)))

	return prediction_fs


