## utils.py -- end-to-end differentable text-to-speech
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import os
import sys
sys.path.append("DeepSpeech")

import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from functools import reduce # used in gather_nd

# Okay, so this is ugly. We don't want DeepSpeech to crash
# when we haven't built the language model.
# So we're just going to monkeypatch TF and make it a no-op.
# Sue me.
tf.load_op_library = lambda x: x
tmp = os.path.exists
os.path.exists = lambda x: True
import DeepSpeech
os.path.exists = tmp

def compute_mfcc(audio, **kwargs):
    """
    Compute the MFCC for a given audio waveform. This is
    identical to how DeepSpeech does it, but does it all in
    TensorFlow so that we can differentiate through it.
    """

    batch_size, size = audio.get_shape().as_list()
    audio = tf.cast(audio, tf.float32)

    # 1. Pre-emphasizer, a high-pass filter
    # => b,t+1000, concat along axis=1, this is 差分
    audio = tf.concat((audio[:, :1], audio[:, 1:] - 0.97*audio[:, :-1], np.zeros((batch_size,1000),dtype=np.float32)), 1)

    # 2. windowing into frames of 320 samples, overlapping
    # => b,?,400
    windowed = tf.stack([audio[:, i:i+400] for i in range(0,size-320,160)],1) # windowsize=400, window_gap=160

    # 3. Take the FFT to convert to frequency space
    # => b,?,257 each window/frame turned to 512 coefs
    ffted = tf.spectral.rfft(windowed, [512])
    ffted = 1.0 / 512 * tf.square(tf.abs(ffted))

    # 4. Compute the Mel windowing of the FFT
    # => b,?
    energy = tf.reduce_sum(ffted,axis=2)+1e-30
    filters = np.load("filterbanks.npy").T
    # => b,?,257 * b,257,26 => b,?,26 this keeps the outer dimension and do the 2dim mult inner
    feat = tf.matmul(ffted, np.array([filters]*batch_size,dtype=np.float32))+1e-30

    # 5. Take the DCT again, because why not
    # b,?,26
    feat = tf.log(feat)
    feat = tf.spectral.dct(feat, type=2, norm='ortho')[:,:,:26]

    # 6. Amplify high frequencies for some reason
    _,nframes,ncoeff = feat.get_shape().as_list()
    n = np.arange(ncoeff)
    lift = 1 + (22/2.)*np.sin(np.pi*n/22)
    feat = lift*feat
    width = feat.get_shape().as_list()[1]

    # 7. And now stick the energy next to the features
    # b,?,26 (1+25)
    feat = tf.concat((tf.reshape(tf.log(energy),(-1,width,1)), feat[:, :, 1:]), axis=2)
    
    return feat

                                          
def get_logits(new_input, length, first=[]):
    """
    Compute the logits for a given waveform.

    First, preprocess with the TF version of MFC above,
    and then call DeepSpeech on the features.
    """

    # We need to init DeepSpeech the first time we're called
    if first == []:
        first.append(False)
        # Okay, so this is ugly again.
        # We just want it to not crash.
        tf.app.flags.FLAGS.alphabet_config_path = "DeepSpeech/data/alphabet.txt"
        DeepSpeech.initialize_globals()

    batch_size = new_input.get_shape()[0]

    # 1. Compute the MFCCs for the input audio
    # TODO why differentiable
    # (this is differentable with our implementation above)
    # TODO why this shape?
    empty_context = np.zeros((batch_size, 9, 26), dtype=np.float32)
    new_input_to_mfcc = compute_mfcc(new_input)[:, ::2]
    features = tf.concat((empty_context, new_input_to_mfcc, empty_context), 1)

    # 2. We get to see 9 frames at a time to make our decision,
    # so concatenate them together.
    features = tf.reshape(features, [new_input.get_shape()[0], -1])
    features = tf.stack([features[:, i:i+19*26] for i in range(0,features.shape[1]-19*26+1,26)],1)
    features = tf.reshape(features, [batch_size, -1, 19*26])

    # 3. Whiten the data
    mean, var = tf.nn.moments(features, axes=[0,1,2])
    features = (features-mean)/(var**.5)

    # 4. Finally we process it with DeepSpeech
    logits = DeepSpeech.BiRNN(features, length, [0]*10)

    return logits

# gather_nd is taken from https://github.com/tensorflow/tensorflow/issues/206#issuecomment-229678962
#
# Unfortunately we can't just use tf.gather_nd because it does not have gradients
# implemented yet, so we need this workaround.
#
def gather_nd(params, indices, shape):
    rank = len(shape)
    flat_params = tf.reshape(params, [-1])
    multipliers = [reduce(lambda x, y: x*y, shape[i+1:], 1) for i in range(0, rank)]
    indices_unpacked = tf.unstack(tf.transpose(indices, [rank - 1] + list(range(0, rank - 1))))
    flat_indices = sum([a*b for a,b in zip(multipliers, indices_unpacked)])
    return tf.gather(flat_params, flat_indices)


# ctc_label_dense_to_sparse is taken from https://github.com/tensorflow/tensorflow/issues/1742#issuecomment-205291527
#
# The CTC implementation in TensorFlow needs labels in a sparse representation,
# but sparse data and queues don't mix well, so we store padded tensors in the
# queue and convert to a sparse representation after dequeuing a batch.
#
def ctc_label_dense_to_sparse(labels, label_lengths, batch_size):
    # The second dimension of labels must be equal to the longest label length in the batch
    correct_shape_assert = tf.assert_equal(tf.shape(labels)[1], tf.reduce_max(label_lengths))
    with tf.control_dependencies([correct_shape_assert]):
        labels = tf.identity(labels)

    label_shape = tf.shape(labels)
    num_batches_tns = tf.stack([label_shape[0]])
    max_num_labels_tns = tf.stack([label_shape[1]])
    def range_less_than(previous_state, current_input):
        return tf.expand_dims(tf.range(label_shape[1]), 0) < current_input

    init = tf.cast(tf.fill(max_num_labels_tns, 0), tf.bool)
    init = tf.expand_dims(init, 0)
    dense_mask = tf.scan(range_less_than, label_lengths, initializer=init, parallel_iterations=1)
    dense_mask = dense_mask[:, 0, :]

    label_array = tf.reshape(tf.tile(tf.range(0, label_shape[1]), num_batches_tns),
          label_shape)
    label_ind = tf.boolean_mask(label_array, dense_mask)

    batch_array = tf.transpose(tf.reshape(tf.tile(tf.range(0, label_shape[0]), max_num_labels_tns), tf.reverse(label_shape, [0])))
    batch_ind = tf.boolean_mask(batch_array, dense_mask)

    indices = tf.transpose(tf.reshape(tf.concat([batch_ind, label_ind], 0), [2, -1]))
    shape = [batch_size, tf.reduce_max(label_lengths)]
    vals_sparse = gather_nd(labels, indices, shape)

    return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64(label_shape))
