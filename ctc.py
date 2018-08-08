# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 18:09:17 2018

@author: Hazekiah
"""
import numpy as np
import tensorflow as tf
def ctc(phrase, logits):
    # target : string
    # logits : n_frame,1,distribution
    target = [ord(x.lower())-ord('a') for x in phrase]
    n_vocab = logits.shape[-1]
    blk_idx = n_vocab-1
    l = []
    for ch in target:
        l.extend([blk_idx,ch])
    l.append(blk_idx)
    n_l = len(l)
    
    n_frame = logits.shape[0]
    ##=======
    # a[t,s] = a[t-1,s],a[t-1,s-1],a[t-1,s-2]
    # s : 2 to n_l
    # t : 1 to n_frame
    # init : t=0, s=0,1
    ##=======
    ##======= trick
    # eps = 1e-16
    # log
    eps = 1e-16
    with tf.Session():
        tmp = tf.nn.softmax(logits)
        probas = tmp.eval()
    a = np.zeros((2,n_l), dtype=np.float32)
    b = np.zeros((n_frame,n_l), dtype=np.float32)
    for s in range(2):
        a[0,s] = probas[0,0,l[s]]
        b[0,s] = probas[0, 0, l[s]]
    for t in range(1,n_frame):
        # avail = np.sum(a[0,:]>0)
        for s in range(2,n_l):
            target_ch = l[s]

            base = b[t-1,s] + b[t-1,s - 1]
            if target_ch != blk_idx and target_ch != l[s - 2]:  # not blank
                base += b[t-1,s - 2]
            b[t,s] = base * probas[t, 0, target_ch]
            print(('b[{t},{s}]={res}'.format(t=t, s=s, res=b[t,s])))
            if s<n_l-2*(n_frame-(t+1))-1-1:
                continue
            base = a[0,s]+a[0,s-1]
            if target_ch!=blk_idx and target_ch!=l[s-2]: # not blank
                base += a[0,s-2]
            if base==0:
                continue
            a[1,s] = base*probas[t,0,target_ch]
            print(('a[{t},{s}]={res}'.format(t=t,s=s,res=a[1,s])))
        a[0,:] = a[1,:]
        a[1,:] = np.zeros(n_l, dtype=np.float32)
    return a[0,n_l-1]+a[0,n_l-2]


def tf_ctc_loss(phrase, logits, is_prob=False):
    """
    CTC loss function.
    phrase - 1xlen(phrase) tensor of int32
    logits - framex1xn_vocab tensor of float32
    return loss value tensor of float32
    params - n x m matrix of n-D probability distributions over m frames.
    seq - sequence of phone id's for given example.
    is_prob - whether params have already passed through a softmax
    Returns objective and gradient.
    """
    # logits frame,b,n_vocab
    blank = logits.shape[-1]-1
    params = tf.squeeze(logits).transpose()
    seq = phrase#np.array([ord(x.lower()) - ord('a') for x in phrase])
    seqLen = seq.shape[0]  # Length of label sequence (# phones)

    numphones = params.shape[0]  # Number of labels
    L = 2 * seqLen + 1  # Length of label sequence with blanks
    T = params.shape[1]  # Length of utterance (time)

    alphas = np.zeros((L, T))
    betas = np.zeros((L, T))

    # Keep for gradcheck move this, assume NN outputs probs
    if not is_prob:
        params = params - np.max(params, axis=0)
        params = np.exp(params)
        params = params / np.sum(params, axis=0)

    # Initialize alphas and forward pass
    alphas[0, 0] = params[blank, 0]
    alphas[1, 0] = params[seq[0], 0]
    c = np.sum(alphas[:, 0])
    alphas[:, 0] = alphas[:, 0] / c
    llForward = np.log(c)

    myloss = 0.

    # params[s,t] at time t the proba of s
    # alphas[s,t] at time t the cumulated proba of matching s
    for t in range(1, T):
        start = max(0, L - 2 * (T - t))
        end = min(2 * t + 2, L)
        for s in range(start, L):
            l = (s - 1) // 2
            # blank
            if s % 2 == 0:
                if s == 0:
                    alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[blank, t]
            # same label twice
            elif s == 1 or seq[l] == seq[l - 1]:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[seq[l], t]
            else:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) \
                               * params[seq[l], t]

        # normalize at current time (prevent underflow)
        c = np.sum(alphas[start:end, t])
        alphas[start:end, t] = alphas[start:end, t] / c
        llForward += np.log(c)
        # alphas[s,t]*V(s,t), V(s,t)=sum(max(params[other,t]-params[s,t],0))
        for tmp in range(L):
            target = params[tmp,t]
            x = [max(other-target,0) for other in params[:,t]]
            v_at_t = np.sum(x)
            myloss += alphas[tmp,t]*v_at_t

    # Initialize betas and backwards pass
    betas[-1, -1] = params[blank, -1]
    betas[-2, -1] = params[seq[-1], -1]
    c = np.sum(betas[:, -1])
    betas[:, -1] = betas[:, -1] / c
    llBackward = np.log(c)
    for t in range(T - 2, -1, -1):
        start = max(0, L - 2 * (T - t))
        end = min(2 * t + 2, L)
        for s in range(end - 1, -1, -1):
            l = (s - 1) // 2
            # blank
            if s % 2 == 0:
                if s == L - 1:
                    betas[s, t] = betas[s, t + 1] * params[blank, t]
                else:
                    betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[blank, t]
            # same label twice
            elif s == L - 2 or seq[l] == seq[l + 1]:
                betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[seq[l], t]
            else:
                betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1] + betas[s + 2, t + 1]) \
                              * params[seq[l], t]

        c = np.sum(betas[start:end, t])
        betas[start:end, t] = betas[start:end, t] / c
        llBackward += np.log(c)

    # Compute gradient with respect to unnormalized input parameters
    grad = np.zeros(params.shape)
    ab = alphas * betas
    for s in range(L):
        # blank
        if s % 2 == 0:
            grad[blank, :] += ab[s, :]
            ab[s, :] = ab[s, :] / params[blank, :]
        else:
            grad[seq[(s - 1) // 2], :] += ab[s, :]
            ab[s, :] = ab[s, :] / (params[seq[(s - 1) // 2], :])
    absum = np.sum(ab, axis=0)

    # Check for underflow or zeros in denominator of gradient
    # llDiff = np.abs(llForward - llBackward)
    # if llDiff > 1e-5 or np.sum(absum == 0) > 0:
    #     print()
    #     "Diff in forward/backward LL : %f" % llDiff
    #     print()
    #     "Zeros found : (%d/%d)" % (np.sum(absum == 0), absum.shape[0])
    #     return -llForward#, grad, True

    grad = params - grad / (params * absum)

    return -llForward, myloss#, grad, False
# TODO decode how? 即便greedy，仍然有一个reduce过程
def ctc_loss(phrase, logits, is_prob=False):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions over m frames.
    seq - sequence of phone id's for given example.
    is_prob - whether params have already passed through a softmax
    Returns objective and gradient.
    """
    # logits frame,b,n_vocab
    blank = logits.shape[-1]-1
    params = np.squeeze(logits).transpose()
    seq = np.array([ord(x.lower()) - ord('a') for x in phrase])
    seqLen = seq.shape[0]  # Length of label sequence (# phones)

    numphones = params.shape[0]  # Number of labels
    L = 2 * seqLen + 1  # Length of label sequence with blanks
    T = params.shape[1]  # Length of utterance (time)

    alphas = np.zeros((L, T))
    betas = np.zeros((L, T))

    # Keep for gradcheck move this, assume NN outputs probs
    if not is_prob:
        params = params - np.max(params, axis=0)
        params = np.exp(params)
        params = params / np.sum(params, axis=0)

    # Initialize alphas and forward pass
    alphas[0, 0] = params[blank, 0]
    alphas[1, 0] = params[seq[0], 0]
    c = np.sum(alphas[:, 0])
    alphas[:, 0] = alphas[:, 0] / c
    llForward = np.log(c)

    myloss = 0.

    # params[s,t] at time t the proba of s
    # alphas[s,t] at time t the cumulated proba of matching s
    for t in range(1, T):
        start = max(0, L - 2 * (T - t))
        end = min(2 * t + 2, L)
        for s in range(start, L):
            l = (s - 1) // 2
            # blank
            if s % 2 == 0:
                if s == 0:
                    alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[blank, t]
            # same label twice
            elif s == 1 or seq[l] == seq[l - 1]:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[seq[l], t]
            else:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) \
                               * params[seq[l], t]

        # normalize at current time (prevent underflow)
        c = np.sum(alphas[start:end, t])
        alphas[start:end, t] = alphas[start:end, t] / c
        llForward += np.log(c)
        # alphas[s,t]*V(s,t), V(s,t)=sum(max(params[other,t]-params[s,t],0))
        for tmp in range(L):
            target = params[tmp,t]
            x = [max(other-target,0) for other in params[:,t]]
            v_at_t = np.sum(x)
            myloss += alphas[tmp,t]*v_at_t

    # Initialize betas and backwards pass
    betas[-1, -1] = params[blank, -1]
    betas[-2, -1] = params[seq[-1], -1]
    c = np.sum(betas[:, -1])
    betas[:, -1] = betas[:, -1] / c
    llBackward = np.log(c)
    for t in range(T - 2, -1, -1):
        start = max(0, L - 2 * (T - t))
        end = min(2 * t + 2, L)
        for s in range(end - 1, -1, -1):
            l = (s - 1) // 2
            # blank
            if s % 2 == 0:
                if s == L - 1:
                    betas[s, t] = betas[s, t + 1] * params[blank, t]
                else:
                    betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[blank, t]
            # same label twice
            elif s == L - 2 or seq[l] == seq[l + 1]:
                betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[seq[l], t]
            else:
                betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1] + betas[s + 2, t + 1]) \
                              * params[seq[l], t]

        c = np.sum(betas[start:end, t])
        betas[start:end, t] = betas[start:end, t] / c
        llBackward += np.log(c)

    # Compute gradient with respect to unnormalized input parameters
    grad = np.zeros(params.shape)
    ab = alphas * betas
    for s in range(L):
        # blank
        if s % 2 == 0:
            grad[blank, :] += ab[s, :]
            ab[s, :] = ab[s, :] / params[blank, :]
        else:
            grad[seq[(s - 1) // 2], :] += ab[s, :]
            ab[s, :] = ab[s, :] / (params[seq[(s - 1) // 2], :])
    absum = np.sum(ab, axis=0)

    # Check for underflow or zeros in denominator of gradient
    # llDiff = np.abs(llForward - llBackward)
    # if llDiff > 1e-5 or np.sum(absum == 0) > 0:
    #     print()
    #     "Diff in forward/backward LL : %f" % llDiff
    #     print()
    #     "Zeros found : (%d/%d)" % (np.sum(absum == 0), absum.shape[0])
    #     return -llForward#, grad, True

    grad = params - grad / (params * absum)

    return -llForward, myloss#, grad, False

from functools import reduce

def gather_nd(params, indices, shape):
    rank = len(shape)
    flat_params = tf.reshape(params, [-1])
    multipliers = [reduce(lambda x, y: x*y, shape[i+1:], 1) for i in range(0, rank)]
    indices_unpacked = tf.unstack(tf.transpose(indices, [rank - 1] + list(range(0, rank - 1))))
    flat_indices = sum([a*b for a,b in zip(multipliers, indices_unpacked)])
    return tf.gather(flat_params, flat_indices)


def ctc_label_dense_to_sparse(labels, label_lengths, batch_size):
    # The second dimension of labels must be equal to the longest label length in the batch

    label_shape = tf.cast(tf.shape(labels), tf.int64)
    correct_shape_assert = tf.assert_equal(label_shape[1], tf.reduce_max(label_lengths)) # int32 vs int64
    with tf.control_dependencies([correct_shape_assert]):
        labels = tf.cast(tf.identity(labels), tf.int32)


    num_batches_tns = tf.stack([label_shape[0]])
    max_num_labels_tns = tf.stack([label_shape[1]])
    def range_less_than(previous_state, current_input):
        return tf.expand_dims(tf.range(label_shape[1]), 0) < current_input # int32 vs int64

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
    vals_sparse = gather_nd(labels, indices, shape) # labels should be of int32

    return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64(label_shape)) # sparseTensor.values should be of int32

def get_logits(frame,b,n_vocab):
    np.random.seed(0)
    logits = np.random.randn(frame,b,n_vocab)
    with tf.Session():
        tmp = tf.nn.softmax(logits)
        logits = tmp.eval()
    return logits

def get_sparse_target(phrase,b):
    tmp = [ord(x.lower())-ord('a') for x in phrase]
    tmp = np.array(tmp)
    tp = tmp.reshape((1,-1))

    tpl = np.array([len(phrase)])
    sparse_target = ctc_label_dense_to_sparse(tp, tpl, b)
    
    return sparse_target

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    zero_out_module = tf.load_op_library('./zero_out.so')
    with self.test_session():
      result = zero_out_module.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])


def compare(phrase):
    frame = 10
    b = 1
    n_vocab = 27
    logits = get_logits(frame,b,n_vocab)
    st = get_sparse_target(phrase,b)
    ctcloss = tf.nn.ctc_loss(st,logits,np.array([frame], dtype=np.int64))  # st should be of dtype int64
    # ctc2 = -np.log(ctc(phrase,logits))
    ctc3, myctc1 = ctc_loss(phrase,logits)
    with tf.Session():
        ctc1 = ctcloss.eval()
    #====================================================================#
    import os
    myctc_module = tf.load_op_library(r'/usr/whz/audio_attack/myctc/myctc.so')
    logits_input = logits[:,0,:]
    phrase_input = np.array([ord(x.lower())-ord('a') for x in phrase])
    with tf.device('/cpu:0'):
        myctc_op = myctc_module.Myctc(phrase_input, logits_input)
        with tf.Session():
            myctc2 = myctc_op.eval()
    #====================================================================#

    return ctc1, ctc3, myctc1, myctc2

if __name__ == '__main__':
    # print((compare('example')))
    tf.test.main()