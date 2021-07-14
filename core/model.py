import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from core.utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1,l2,l1_l2
import tensorflow as tf
import tempfile
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
import sys
class Model():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def build_model(self, configs):
		timer = Timer()
		timer.start()

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None
			L1 = layer['L1'] if 'L1' in layer else None
			L2 = layer['L2'] if 'L1' in layer else None
			reg=None
			print("boop")
			if ((L1!=None) and (L2!=None)):
				reg=l1_l2(l1=L1,l2=L2)
			elif (L1 != None):
				reg=l1(l1=L1)
			elif (L2 != None):
				reg=l2(l2=L2)

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation,kernel_regularizer=reg))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq,kernel_regularizer=reg))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')
		timer.stop()
		

	def train(self, x, y, epochs, batch_size, save_dir):
		configs = json.load(open('config.json', 'r'))

		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		if 'trained_model_name' in configs['data']:
			save_fname = os.path.join(save_dir,configs['data']['trained_model_name'] ,'trained_at_%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		else:
			save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
	
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
		self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
		
		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def predict_point_by_point(self, data):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		print('[Model] Predicting Point-by-Point...')
		predicted = self.model.predict(data)
		predicted = np.reshape(predicted, (predicted.size,))
		return predicted

	def predict_sequences_multiple(self, data, window_size, prediction_len):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		print('[Model] Predicting Sequences Multiple...')
		prediction_seqs = []
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			prediction_seqs.append(predicted)
		return prediction_seqs

	def predict_sequence_full(self, data, window_size):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
		print('[Model] Predicting Sequences Full...')
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		return predicted
	

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

def sparsity_pruning(configs,data,model,save_dir):
	prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
	

	save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), "pruned"))
	
	# Compute end step to finish pruning after 2 epochs.
	batch_size = configs["training"]["batch_size"]
	epochs = configs["training"]["epochs"]
	validation_split = configs["training"]["validation_split"]
	x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    ) 
	num_data_points = x.shape[0] * (1 - validation_split)
	end_step = np.ceil(num_data_points / batch_size).astype(np.int32) * epochs

	# Define model for pruning.
	pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=configs["pruning_parameters"]["initial_sparsity"],
                                                               final_sparsity=configs["pruning_parameters"]["final_sparsity"],
                                                               begin_step=0,
                                                               end_step=end_step)
	}						
	model_for_pruning = prune_low_magnitude(model, **pruning_params)

	# `prune_low_magnitude` requires a recompile.
	model_for_pruning.compile(optimizer=configs["model"]["optimizer"],
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

	model_for_pruning.summary()
	logdir = tempfile.mkdtemp()

	callbacks = [
		tfmot.sparsity.keras.UpdatePruningStep(),
		tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
	]
	
	model_for_pruning.fit(x, y,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)
	try:
		model_for_pruning.save(save_fname)
	except:
		model_for_pruning.save(save_dir)

	return save_fname


def small_model(configs,data,model,save_dir):
	#if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

	#fname=sparsity_pruning(configs,data,model,save_dir)
	#model=load_model(fname)
	prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
	

	save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), "pruned"))
	
	# Compute end step to finish pruning after 2 epochs.
	batch_size = configs["training"]["batch_size"]
	epochs = configs["training"]["epochs"]
	validation_split = configs["training"]["validation_split"]
	x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    ) 
	num_data_points = x.shape[0] * (1 - validation_split)
	end_step = np.ceil(num_data_points / batch_size).astype(np.int32) * epochs

	# Define model for pruning.
	pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=configs["pruning_parameters"]["initial_sparsity"],
                                                               final_sparsity=configs["pruning_parameters"]["final_sparsity"],
                                                               begin_step=0,
                                                               end_step=end_step)
	}						
	model_for_pruning = prune_low_magnitude(model, **pruning_params)

	# `prune_low_magnitude` requires a recompile.
	model_for_pruning.compile(optimizer=configs["model"]["optimizer"],
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

	model_for_pruning.summary()
	logdir = tempfile.mkdtemp()

	callbacks = [
		tfmot.sparsity.keras.UpdatePruningStep(),
		tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
	]
	
	model_for_pruning.fit(x, y,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)
	model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

	_, pruned_keras_file = tempfile.mkstemp('.h5')
	tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
	print('Saved pruned Keras model to:', pruned_keras_file)

	converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
	pruned_tflite_model = converter.convert()
	
	_, pruned_tflite_file = tempfile.mkstemp('.tflite')
	with open(pruned_tflite_file, 'wb') as f:
		f.write(pruned_tflite_model)

	print('Saved pruned TFLite model to:', pruned_tflite_file)
	return pruned_tflite_file

def very_small_model(configs,data,model,save_dir):
	old=sys.stdout
	sys.stdout=open("pruning-engine.txt",'w')
	timer = Timer()
	timer.start()
	prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
	

	save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), "pruned"))
	
	# Compute end step to finish pruning after 2 epochs.
	batch_size = configs["training"]["batch_size"]
	epochs = configs["training"]["epochs"]
	validation_split = configs["training"]["validation_split"]
	x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    ) 
	num_data_points = x.shape[0] * (1 - validation_split)
	end_step = np.ceil(num_data_points / batch_size).astype(np.int32) * epochs

	# Define model for pruning.
	pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=configs["pruning_parameters"]["initial_sparsity"],
                                                               final_sparsity=configs["pruning_parameters"]["final_sparsity"],
                                                               begin_step=0,
                                                               end_step=end_step)
	}						
	model_for_pruning = prune_low_magnitude(model, **pruning_params)

	# `prune_low_magnitude` requires a recompile.
	model_for_pruning.compile(optimizer=configs["model"]["optimizer"],
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

	model_for_pruning.summary()
	logdir = tempfile.mkdtemp()

	callbacks = [
		tfmot.sparsity.keras.UpdatePruningStep(),
		tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
	]
	
	model_for_pruning.fit(x, y,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)
	model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

	_, pruned_keras_file = tempfile.mkstemp('.h5')
	tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
	print('Saved pruned Keras model to:', pruned_keras_file)

	converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
	pruned_tflite_model = converter.convert()
	
	_, pruned_tflite_file = tempfile.mkstemp('.tflite')
	with open(pruned_tflite_file, 'wb') as f:
		f.write(pruned_tflite_model)

	print('Saved pruned TFLite model to:', pruned_tflite_file)
	converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	quantized_and_pruned_tflite_model = converter.convert()

	_, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

	with open(quantized_and_pruned_tflite_file, 'wb') as f:
		f.write(quantized_and_pruned_tflite_model)

	print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)

	#print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
	print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file)))
	timer.stop()
	sys.stdout.close()
	sys.stdout=old