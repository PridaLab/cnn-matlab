import tensorflow.keras.backend as K
import tensorflow.keras as kr
import numpy as np
import os


class ModelClass:

	def __init__ (self, downsampled_fs, randomize_weights=False, weights=[1.]):
		self.downsampled_fs = downsampled_fs
		self.num_classes = 1
		self.loss_weights = np.array([1.])

		# Optimizer and regularizer by default
		self.initial_learning_rate = 0.001
		self.optimizer = kr.optimizers.Adam()
		self.regularizer = None


	def time_scheduler(self, epoch, lr):
		return lr * 1/(1 + self.decay * epoch)

	def drop_scheduler(self, epoch, lr):
		return self.initial_learning_rate * self.drop_rate**np.floor(epoch / self.epoch_drop)

	def half_scheduler(self, epoch, lr):
		if epoch % self.epoch_drop == 0:
			return lr / 2.

		return lr
			


	def set_optimizer(self, optimizer_name, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
		self.initial_learning_rate = learning_rate

		if optimizer_name == "Adam":
			self.optimizer = kr.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=False)
		#elif optimizer_name == "SGD":
		#	self.optimizer = kr.optimizers.SGD(nesterov=True, learning_rate=learning_rate)
		elif optimizer_name == "AMSGrad":
			self.optimizer = kr.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=True)

		return


	def set_regularizer(self, regularizer_name, value):
		if regularizer_name == 'l1':
			self.regularizer = kr.regularizers.l1(value)
		elif regularizer_name == 'l2':
			self.regularizer = kr.regularizers.l2(value)
		elif regularizer_name == 'l1_l2':
			self.regularizer = kr.regularizers.l1_l2(value)

		return



	def set_architecture(self, window_size_ms, stride_eff_ms, num_filters_factor):
		self.window_size = window_size_ms * self.downsampled_fs
		kernel_size = []
		stride = []
		padding = []
		layer_type = []
		num_filters = []

		output_size = self.num_classes

		if self.window_size == 96.0:
			kernel_size  = [   3, 1,    2, 1,    2, 1,    2, 1,    2, 1,    2, 1] 
			stride       = [   3, 1,    2, 1,    2, 1,    2, 1,    2, 1,    2, 1]
			padding      = [   0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0] 
			layer_type   = ['c','c', 'c','c', 'c','c', 'c','c', 'c','c', 'c','c'] 
			num_filters  = [  32*num_filters_factor,16*num_filters_factor,   64*num_filters_factor,32*num_filters_factor,  128*num_filters_factor,64*num_filters_factor, 256*num_filters_factor,128*num_filters_factor, 512*num_filters_factor,256*num_filters_factor, 1024*num_filters_factor,output_size]
		elif self.window_size == 40.0:
			kernel_size  = [   5, 1,    2, 1,    2, 1,    2, 1] 
			stride       = [   5, 1,    2, 1,    2, 1,    2, 1]
			padding      = [   0, 0,    0, 0,    0, 0,    0, 0] 
			layer_type   = ['c','c', 'c','c', 'c','c', 'c','c'] 
			num_filters  = [  32*num_filters_factor,16*num_filters_factor,   64*num_filters_factor,32*num_filters_factor,  128*num_filters_factor,64*num_filters_factor,   256*num_filters_factor,output_size]
		elif self.window_size == 16.0:
			kernel_size  = [   2, 1,    2, 1,    2, 1,    2, 1] 
			stride       = [   2, 1,    2, 1,    2, 1,    2, 1]
			padding      = [   0, 0,    0, 0,    0, 0,    0, 0] 
			layer_type   = ['c','c', 'c','c', 'c','c', 'c','c'] 
			num_filters  = [  32*num_filters_factor,16*num_filters_factor,   64*num_filters_factor,32*num_filters_factor,  128*num_filters_factor,64*num_filters_factor,   256*num_filters_factor,output_size]
		elif self.window_size == 111.0: # 5.0 
			kernel_size  = [   5, 1] 
			stride       = [   5, 1]
			padding      = [   0, 0] 
			layer_type   = ['c','c'] 
			num_filters  = [  32*num_filters_factor,output_size]
		elif self.window_size == 222.0: # 5.0
			kernel_size  = [   5, 1, 1] 
			stride       = [   5, 1, 1]
			padding      = [   0, 0, 0] 
			layer_type   = ['c','c','c'] 
			num_filters  = [  32*num_filters_factor,16*num_filters_factor,output_size]
		elif self.window_size == 333.0: # 40.0
			kernel_size  = [   8, 1, 5, 1] 
			stride       = [   8, 1, 5, 1]
			padding      = [   0, 0, 0, 0] 
			layer_type   = ['c','c', 's','c'] 
			num_filters  = [  16*num_filters_factor,8*num_filters_factor, 32*num_filters_factor,output_size]
		elif self.window_size == 320.0:
			kernel_size  = [   8, 1, 5, 1,    2, 1,    2, 1,    2, 1] 
			stride       = [   8, 1, 5, 1,    2, 1,    2, 1,    2, 1]
			padding      = [   0, 0, 0, 0,    0, 0,    0, 0,    0, 0] 
			layer_type   = ['c','c', 'c','c', 'c','c', 'c','c', 'c','c'] 
			num_filters  = [  16*num_filters_factor,8*num_filters_factor, 32*num_filters_factor,16*num_filters_factor,   64*num_filters_factor,32*num_filters_factor,  128*num_filters_factor,64*num_filters_factor,   256*num_filters_factor,output_size]


		return kernel_size, stride, padding, layer_type, num_filters


	# https://datascience.stackexchange.com/questions/58735/weighted-binary-cross-entropy-loss-keras-implementation
	def weighted_bce(self, y_true, y_pred):
		weights = (y_true * self.loss_weights[0]) + 1.
		bce = K.binary_crossentropy(y_true, y_pred)
		weighted_bce = K.mean(bce * weights)
		return weighted_bce


	def build_model(self, layer_type, num_filters, kernel_size, stride, padding, input_shape):
		# Make CNN
		self.model = kr.models.Sequential()

		for layer in range(len(layer_type)):

			# Layer characteristics
			n = num_filters[layer]
			k = kernel_size[layer]
			s = stride[layer]
			p = 'valid' if padding[layer]==0 else 'same'

			# First layer: CONV with specific input shape
			if layer == 0:
				# Each convolutional layer is followed by a Batch Normalization layer, and then an activation layer
				self.model.add(kr.layers.Conv1D(filters=n, kernel_size=k, strides=s, padding=p, input_shape=input_shape, kernel_regularizer=self.regularizer))
				self.model.add(kr.layers.BatchNormalization())
				self.model.add(kr.layers.LeakyReLU(alpha=0.1))

			# Last layer: CONV without BatchNorm, followed by a Softmax activation
			elif layer==(len(layer_type)-1):
				self.model.add(kr.layers.Dense(n, activation="sigmoid"))
				#self.model.add(kr.layers.Conv1D(filters=n, kernel_size=k, strides=s, padding=p, activation='linear', input_shape=input_shape, kernel_regularizer=self.regularizer))
				#self.model.add(kr.layers.GlobalAveragePooling1D())
				#self.model.add(kr.layers.LeakyReLU(alpha=0.1))
				#self.model.add(kr.layers.Softmax())

			# Rest of layers
			else:
				# Convolutional layer
				if layer_type[layer]=='c':
					# Each convolutional layer is followed by a Batch Normalization layer, and then an activation layer
					self.model.add(kr.layers.Conv1D(filters=n, kernel_size=k, strides=s, padding=p, kernel_regularizer=self.regularizer))
					self.model.add(kr.layers.BatchNormalization())
					self.model.add(kr.layers.LeakyReLU(alpha=0.1))
				# Max-Pool layer
				if layer_type[layer]=='m':
					self.model.add(kr.layers.MaxPooling1D(pool_size=k, strides=s, padding=p))
				# Solo convolutional layer
				if layer_type[layer]=='s':
					self.model.add(kr.layers.Conv1D(filters=n, kernel_size=k, strides=s, padding=p, kernel_regularizer=self.regularizer))

		# Compile network
		self.model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=self.optimizer)	# fit


	def load_model(self, model_file):
		self.model = kr.models.load_model(model_file, compile=False)
		self.model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=self.optimizer)


	def save_model(self, path, tag):
		self.model.save(os.path.join(path,"cnn_model"+tag))

	
	def model_summary(self):
		self.model.summary()

	
	def model_fit(self, X, y, num_epochs, batch_size, verbose, do_shuffle, decay_type, save_training_path=''):
		verbose = 2 if verbose == 1 else 0 # verbose == 2 is just one line per epoch


		callbacks = [kr.callbacks.TerminateOnNaN()]
		if decay_type == 'time':
			self.decay = self.initial_learning_rate / num_epochs
			learning_rate_callback = kr.callbacks.LearningRateScheduler(self.time_scheduler)
			callbacks.append(learning_rate_callback)
		elif decay_type == 'drop':
			self.drop_rate = 0.5
			self.epoch_drop = 10.
			learning_rate_callback = kr.callbacks.LearningRateScheduler(self.drop_scheduler)
			callbacks.append(learning_rate_callback)
		elif decay_type == 'half':
			self.epoch_drop = 100.
			learning_rate_callback = kr.callbacks.LearningRateScheduler(self.half_scheduler)
			callbacks.append(learning_rate_callback)

		# Save weights of all epochs of training
		if len(save_training_path) > 0:
			save_training_callback = kr.callbacks.ModelCheckpoint(os.path.join(save_training_path,"weights_{epoch:04d}.hdf5"))
			callbacks.append(save_training_callback)


		return self.model.fit(X, y, epochs=num_epochs, batch_size=batch_size, verbose=verbose, shuffle=do_shuffle, callbacks=[callbacks])

	
	def model_evaluate(self, X, y, batch_size, verbose):
		return self.model.evaluate(X, y, batch_size=batch_size, verbose=verbose)

	
	def model_predict(self, X, batch_size=1, verbose=1):
		return self.model.predict(X, batch_size=batch_size, verbose=verbose)

	def get_prediction_times(self, X_true, y_pred, threshold=0.5, threshold_low = -1, i_class=0):

		n_chunks = np.shape(y_pred)[0]
		n_windows = np.shape(y_pred)[1]
		pts_per_chunk = np.shape(X_true)[1]
		pts_per_window = int(pts_per_chunk / n_windows)

		if threshold_low == -1:
			threshold_low = threshold / 2


		# We get events over high threshold
		events_times_high = []

		for i_chunk in range(n_chunks):

			# Generate probability signals
			pred_signal = np.zeros(pts_per_chunk)

			for i_window in range(n_windows):
				first = i_window*pts_per_window
				last = (i_window+1)*pts_per_window
				pred_signal[first:last] = y_pred[i_chunk, i_window, i_class]
				# Get the max probability from all classes
				#pred_signal[first:last] = np.max(y_pred[i_chunk, i_window])


			# Create masks with True when the signals cross the threshold
			mask_pred = np.diff(1 * (pred_signal >= threshold) != 0)

			# Get the times (in points) where the signals cross the threshold
			pred_times = np.argwhere(mask_pred == True).flatten()

			# Get the directions of the threshold crossing (neg when ascending, pos when descending)
			pred_directions = pred_signal[pred_times] - pred_signal[pred_times+1]

			# Get the intervals starting and ending times
			pred_inis = pred_times[pred_directions < 0]
			pred_ends = pred_times[pred_directions > 0]
			if len(pred_ends) < len(pred_inis):
				# If there is one more ini than end
				pred_ends = np.append(pred_ends, pts_per_chunk-1)
			elif len(pred_ends) > len(pred_inis):
				# If there is one more end than ini
				pred_inis = np.insert(pred_inis, 0, 0)
			# Make a (# events)x2 array
			pred_inis = np.reshape(pred_inis, (-1, 1))
			pred_ends = np.reshape(pred_ends, (-1, 1))
			iniend = np.append(pred_inis, pred_ends, axis=1)
			
			# events_times_high[chunk, class, event] -> even if there is no class in here
			#events_times_high.append([])
			#events_times_high[-1].append(iniend/self.downsampled_fs)

			events_times_high.append(iniend/self.downsampled_fs)


		# We get events over low threshold
		events_times_low = []

		for i_chunk in range(n_chunks):

			# Generate probability signals
			pred_signal = np.zeros(pts_per_chunk)

			for i_window in range(n_windows):
				first = i_window*pts_per_window
				last = (i_window+1)*pts_per_window
				pred_signal[first:last] = y_pred[i_chunk, i_window, i_class]
				# Get the max probability from all classes
				#pred_signal[first:last] = np.max(y_pred[i_chunk, i_window])


			# Create masks with True when the signals cross the threshold
			mask_pred = np.diff(1 * (pred_signal >= threshold_low) != 0)

			# Get the times (in points) where the signals cross the threshold
			pred_times = np.argwhere(mask_pred == True).flatten()

			# Get the directions of the threshold crossing (neg when ascending, pos when descending)
			pred_directions = pred_signal[pred_times] - pred_signal[pred_times+1]

			# Get the intervals starting and ending times
			pred_inis = pred_times[pred_directions < 0]
			pred_ends = pred_times[pred_directions > 0]
			if len(pred_ends) < len(pred_inis):
				# If there is one more ini than end
				pred_ends = np.append(pred_ends, pts_per_chunk-1)
			elif len(pred_ends) > len(pred_inis):
				# If there is one more end than ini
				pred_inis = np.insert(pred_inis, 0, 0)
			# Make a (# events)x2 array
			pred_inis = np.reshape(pred_inis, (-1, 1))
			pred_ends = np.reshape(pred_ends, (-1, 1))
			iniend = np.append(pred_inis, pred_ends, axis=1)
			
			# events_times_low[chunk, class, event] -> even if there is no class in here
			#events_times_low.append([])
			#events_times_low[-1].append(iniend/self.downsampled_fs)

			events_times_low.append(iniend/self.downsampled_fs)


		# We get low events that have a match with high events	
		events_times = []

		for i_chunk in range(n_chunks):
			events_times.append([])
			chunk_times = events_times[-1] 

			for low_event in events_times_low[i_chunk]:
				for high_event in events_times_high[i_chunk]:
					if (low_event[0] <= high_event[0]) and (low_event[1] >= high_event[1]):
						chunk_times.append(low_event)

		return events_times