#talking-heads attention
import random, os, sys
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
# from keras.utils import multi_gpu_model

try:
	from tqdm import tqdm
	from dataloader import TokenList, pad_to_longest
	# for transformer
except: pass

class LayerNormalization(Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape

# It's safe to use a 1-d mask for self-attention
class ScaledDotProductAttention():
	def __init__(self,n_head, attn_dropout=0.1):
		self.dropout = Dropout(attn_dropout)
		self.n_head = n_head
		self.wgt = tf.Variable(initial_value=tf.truncated_normal(shape=[self.n_head, self.n_head]))
		self.wgt2tensor = Lambda(lambda x:tf.convert_to_tensor(x))(self.wgt)
		self.weight_layer = Dense(n_head, use_bias=False)
	def __call__(self, q, k, v, mask):   # mask_k or mask_qk  [n_head * batch_size, len, d_k]

		def reshape1(x):
			s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
			x = tf.reshape(x, [ s[0]//self.n_head,self.n_head,s[1],s[2]])
			return x
		def reshape2(x):
			s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
			x = tf.reshape(x, [-1,s[2],s[3]])
			return x
		def eins(x):
			o = tf.einsum('ij,kjmn->kimn',x[0],x[1])
			return o

		temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/temper)([q, k])  # shape=(batch*heads, lq, lk)
		if mask is not None:
			mmask = Lambda(lambda x:(-1e+9)*(1.-K.cast(x, 'float32')))(mask)
			attn = Add()([attn, mmask])		# shape=(batch*heads, lq, lk)
		#attn [n_head * batch_size, len, len]
		#talking-heads
		attn = Lambda(reshape1)(attn)  #[batch_size,n_head, len, len]
		# wm = self.weight_layer(self.wgt)
		# attn = Lambda(eins)([wm,attn])   #[batch_size,n_head, len, len]
		attn = Lambda(reshape2)(attn)       #attn [n_head * batch_size, len, len]
		# #talking-heads end

		attn = Activation('softmax')(attn)
		attn = self.dropout(attn)
		output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
		return output, attn

class MultiHeadAttention():
	# mode 0 - big martixes, faster; mode 1 - more clear implementation
	def __init__(self, n_head, d_model, dropout, mode=0):
		self.mode = mode
		self.n_head = n_head
		self.d_k = self.d_v = d_k = d_v = d_model // n_head
		self.dropout = dropout
		if mode == 0:
			self.qs_layer = Dense(n_head*d_k, use_bias=False)
			self.ks_layer = Dense(n_head*d_k, use_bias=False)
			self.vs_layer = Dense(n_head*d_v, use_bias=False)
		elif mode == 1:
			self.qs_layers = []
			self.ks_layers = []
			self.vs_layers = []
			for _ in range(n_head):
				self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
		self.attention = ScaledDotProductAttention(n_head)
		self.w_o = TimeDistributed(Dense(d_model))

	def __call__(self, q, k, v, mask=None):
		'''
		:param q: [batch_size,lq,dmodel]
		:param k: [batch_size,lk,dmodel]
		:param v: [batch_size,lk,dmodel]
		:param mask: encoder_self_mask:[batch_size,lq]  decoder_self_mask:[batch_size,lq,lq]
					decoder_encoder_mask:[batch_size,lq,lk]
		:return:[batch_size,lq,dmodel]
		'''

		d_k, d_v = self.d_k, self.d_v
		n_head = self.n_head

		if self.mode == 0:
			qs = self.qs_layer(q)
			ks = self.ks_layer(k)
			vs = self.vs_layer(v)

			def reshape1(x):
				s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
				x = tf.reshape(x, [s[0], s[1], n_head, s[2]//n_head])
				x = tf.transpose(x, [2, 0, 1, 3])  
				x = tf.reshape(x, [-1, s[1], s[2]//n_head])  # [n_head * batch_size, len_q, d_k]
				return x
			qs = Lambda(reshape1)(qs)	# [n_head*batch_size, lq, dk]
			ks = Lambda(reshape1)(ks)	# [n_head*batch_size, lk, dk]
			vs = Lambda(reshape1)(vs)	# [n_head*batch_size, lk, dv]

			if mask is not None:
				mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
			head, attn = self.attention(qs, ks, vs, mask=mask)  
				
			def reshape2(x):
				s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
				x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
				x = tf.transpose(x, [1, 2, 0, 3])
				x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
				return x
			head = Lambda(reshape2)(head)


		outputs = self.w_o(head)
		outputs = Dropout(self.dropout)(outputs)
		return outputs, attn

class PositionwiseFeedForward():
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
		self.w_2 = Conv1D(d_hid, 1)
		self.layer_norm = LayerNormalization()
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		output = self.w_1(x) 
		output = self.w_2(output)
		output = self.dropout(output)
		output = Add()([output, x])
		return self.layer_norm(output)

# class EncoderLayer():
# 	def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
# 		self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
# 		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
# 		self.norm_layer = LayerNormalization()
# 	def __call__(self, enc_input, mask=None):
# 		output = self.norm_layer(enc_input)
# 		output, slf_attn = self.self_att_layer(output, output, output, mask=mask)
# 		output = Add()([enc_input, output])
# 		output = self.pos_ffn_layer(output)
# 		return output, slf_attn

class EncoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, dropout=0.3):
		self.self_att_layer = InteractingHeads(n_head, d_model, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
		self.norm_layer = LayerNormalization()
	def __call__(self, enc_input, mask=None):
		output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
		output = self.norm_layer(Add()([enc_input, output]))
		output = self.pos_ffn_layer(output)
		return output, slf_attn

class DecoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
		self.self_att_layer = InteractingHeads(n_head, d_model, dropout=dropout)
		self.enc_att_layer  = InteractingHeads(n_head, d_model, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
		self.norm_layer1 = LayerNormalization()
		self.norm_layer2 = LayerNormalization()
	def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None, dec_last_state=None):
		if dec_last_state is None: dec_last_state = dec_input
		output, slf_attn = self.self_att_layer(dec_input, dec_last_state, dec_last_state, mask=self_mask)
		x = self.norm_layer1(Add()([dec_input, output]))
		output, enc_attn = self.enc_att_layer(x, enc_output, enc_output, mask=enc_mask)
		x = self.norm_layer2(Add()([x, output]))
		output = self.pos_ffn_layer(x)
		return output, slf_attn, enc_attn

def GetPosEncodingMatrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
		if pos != 0 else np.zeros(d_emb) 
			for pos in range(max_len)
			])
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
	return pos_enc

def GetPadMask(q, k):
	'''
	shape: [B, Q, K]
	'''
	ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
	mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
	mask = K.batch_dot(ones, mask, axes=[2,1])
	return mask

def GetSubMask(s):
	'''
	shape: [B, Q, K], lower triangle because the i-th row should have i 1s.
	'''
	len_s = tf.shape(s)[1]
	bs = tf.shape(s)[:1]
	mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
	return mask

class SelfAttention():
	def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.3):
		self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]
	def __call__(self, src_emb, src_seq, return_att=False, active_layers=999):
		if return_att: atts = []
		mask = Lambda(lambda x:K.cast(K.greater(x, 0), 'float32'))(src_seq)
		x = src_emb		
		for enc_layer in self.layers[:active_layers]:
			x, att = enc_layer(x, mask)
			if return_att: atts.append(att)
		return (x, atts) if return_att else x

class Decoder():
	def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
		self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]
	def __call__(self, tgt_emb, tgt_seq, src_seq, enc_output, return_att=False, active_layers=999):
		x = tgt_emb
		self_pad_mask = Lambda(lambda x:GetPadMask(x, x))(tgt_seq)
		self_sub_mask = Lambda(GetSubMask)(tgt_seq)
		self_mask = Lambda(lambda x:K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
		enc_mask = Lambda(lambda x:GetPadMask(x[0], x[1]))([tgt_seq, src_seq])
		if return_att: self_atts, enc_atts = [], []
		print(enc_mask.shape)
		for dec_layer in self.layers[:active_layers]:
			x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
			if return_att: 
				self_atts.append(self_att)
				enc_atts.append(enc_att)
		return (x, self_atts, enc_atts) if return_att else x

class DecoderPerStep(Layer):
	def __init__(self, decoder):
		super().__init__()
		self.layers = decoder.layers
	def call(self, inputs):
		(x, src_seq, enc_output), tgt_embs = inputs[:3], inputs[3:]
		enc_mask = K.cast(K.greater(src_seq, 0), 'float32')
		llen = tf.shape(tgt_embs[0])[1]
		col_mask = K.cast(K.equal(K.cumsum(K.ones_like(tgt_embs[0], dtype='int32'), axis=1), llen), dtype='float32')
		rs = [x]
		for i, dec_layer in enumerate(self.layers):
			tgt_emb = tgt_embs[i] + x * col_mask
			x, _, _ = dec_layer(x, enc_output, enc_mask=enc_mask, dec_last_state=tgt_emb)
			rs.append(x)
		return rs
	def compute_output_shape(self, ishape):
		return [ishape[0] for _ in range(len(self.layers)+1)]

class ReadoutDecoderCell(Layer):
	def __init__(self, o_word_emb, pos_emb, decoder, target_layer, **kwargs):
		self.o_word_emb = o_word_emb
		self.pos_emb = pos_emb
		self.decoder = decoder
		self.target_layer = target_layer
		super().__init__(**kwargs)
	def call(self, inputs, states, constants, training=None):
		(tgt_curr_input, tgt_pos_input, dec_mask), dec_output = states[:3], list(states[3:])
		enc_output, enc_mask = constants

		time = K.max(tgt_pos_input)
		col_mask = K.cast(K.equal(K.cumsum(K.ones_like(dec_mask), axis=1), time), dtype='int32')
		dec_mask = dec_mask + col_mask

		tgt_emb = self.o_word_emb(tgt_curr_input)
		if self.pos_emb: tgt_emb = tgt_emb + self.pos_emb(tgt_pos_input, pos_input=True)

		x = tgt_emb
		xs = []
		cc = K.cast(K.expand_dims(col_mask), dtype='float32')
		for i, dec_layer in enumerate(self.decoder.layers):
			dec_last_state = dec_output[i] * (1-cc) + tf.einsum('ijk,ilj->ilk', x, cc)
			x, _, _ = dec_layer(x, enc_output, dec_mask, enc_mask, dec_last_state=dec_last_state)
			xs.append(dec_last_state)

		ff_output = self.target_layer(x)
		out = K.cast(K.argmax(ff_output, -1), dtype='int32')
		return out, [out, tgt_pos_input+1, dec_mask] + xs


def decode_batch_greedy(src_seq, encode_model, decode_model, start_mark, end_mark, max_len=128):
	enc_ret = encode_model.predict_on_batch(src_seq)
	bs = src_seq.shape[0]
	target_one = np.zeros((bs, 1), dtype='int32')
	target_one[:,0] = start_mark
	d_model = decode_model.inputs[-1].shape[-1]
	n_dlayers = len(decode_model.inputs) - 3
	dec_outputs = [np.zeros((bs, 1, d_model)) for _ in range(n_dlayers)]
	ended = [0 for x in range(bs)]
	decoded_indexes = [[] for x in range(bs)]
	for i in range(max_len-1):
		outputs = decode_model.predict_on_batch([target_one, src_seq, enc_ret] + dec_outputs)
		new_dec_outputs, output = outputs[:-1], outputs[-1]
		for dec_output, new_out in zip(dec_outputs, new_dec_outputs): 
			dec_output[:,-1,:] = new_out[:,0,:]
		dec_outputs = [np.concatenate([x, np.zeros_like(new_out)], axis=1) for x in dec_outputs]

		sampled_indexes = np.argmax(output[:,0,:], axis=-1)
		for ii, sampled_index in enumerate(sampled_indexes):
			if sampled_index == end_mark: ended[ii] = 1
			if not ended[ii]: decoded_indexes[ii].append(sampled_index)
		if sum(ended) == bs: break
		target_one[:,0] = sampled_indexes
	return decoded_indexes


def decode_batch_beam_search(src_seq, topk, encode_model, decode_model, start_mark, end_mark, max_len=128, early_stop_mult=5):
	N = src_seq.shape[0]
	src_seq = src_seq.repeat(topk, 0)
	enc_ret = encode_model.predict_on_batch(src_seq)
	bs = src_seq.shape[0]

	target_one = np.zeros((bs, 1), dtype='int32')
	target_one[:,0] = start_mark
	d_model = decode_model.inputs[-1].shape[-1]
	n_dlayers = len(decode_model.inputs) - 3
	dec_outputs = [np.zeros((bs, 1, d_model)) for _ in range(n_dlayers)]

	final_results = []
	decoded_indexes = [[] for x in range(bs)]
	decoded_logps = [0] * bs
	lastks = [1 for x in range(N)]
	bests = {}
	for i in range(max_len):
		outputs = decode_model.predict_on_batch([target_one, src_seq, enc_ret] + dec_outputs)
		new_dec_outputs, output = outputs[:-1], outputs[-1]
		for dec_output, new_out in zip(dec_outputs, new_dec_outputs):
			dec_output[:,-1,:] = new_out[:,0,:]

		dec_outputs = [np.concatenate([x, np.zeros_like(new_out)], axis=1) for x in dec_outputs]

		output = np.exp(output[:,0,:])
		output = np.log(output / np.sum(output, -1, keepdims=True) + 1e-8)

		next_dec_outputs = [x.copy() for x in dec_outputs]
		next_decoded_indexes = [1 for x in range(bs)]

		for ii in range(N):
			base = ii * topk
			cands = []
			for k, wprobs in zip(range(lastks[ii]), output[base:,:]):
				prev = base+k
				if len(decoded_indexes[prev]) > 0 and decoded_indexes[prev][-1] == end_mark: continue
				ind = np.argpartition(wprobs, -topk)[-topk:]
				wsorted = [(k,x) for k,x in zip(ind, wprobs[ind])]
				#wsorted = sorted(list(enumerate(wprobs)), key=lambda x:x[-1], reverse=True)   # slow
				for wid, wp in wsorted[:topk]:
					wprob = decoded_logps[prev]+wp
					if wprob < bests.get(ii, -1e5) * early_stop_mult: continue
					cands.append( (prev, wid, wprob) )
			cands.sort(key=lambda x:x[-1], reverse=True)
			cands = cands[:topk]
			lastks[ii] = len(cands)
			for kk, zz in enumerate(cands):
				prev, wid, wprob = zz
				npos = base+kk
				for k in range(len(next_dec_outputs)):
					next_dec_outputs[k][npos,:,:] = dec_outputs[k][prev]
				target_one[npos,0] = wid
				decoded_logps[npos] = wprob
				next_decoded_indexes[npos] = decoded_indexes[prev].copy()
				next_decoded_indexes[npos].append(wid)
				if wid == end_mark or i==max_len-1:
					final_results.append( (ii, decoded_indexes[prev].copy(), wprob) )
					if ii not in bests or wprob > bests[ii]: bests[ii] = wprob
		if sum(lastks) == 0: break
		dec_outputs = next_dec_outputs
		decoded_indexes = next_decoded_indexes
	return final_results

import keras

class Transformer:
	def __init__(self, i_tokens, o_tokens, len_limit, d_model=256, \
			  d_inner_hid=512, n_head=4, layers=6, dropout=0.3, eps=0.1,\
			  share_word_emb=False):
		self.i_tokens = i_tokens
		self.o_tokens = o_tokens
		self.len_limit = len_limit
		self.d_model = d_model
		self.decode_model = None
		self.readout_model = None
		self.layers = layers
		self.eps = eps
		d_emb = d_model

		self.src_loc_info = True

		d_k = d_v = d_model // n_head
		assert d_k * n_head == d_model and d_v == d_k

		self.label_smoothing = LabelSmoothing(self.eps)

		self.pos_emb = PosEncodingLayer(len_limit, d_emb) if self.src_loc_info else None

		self.emb_dropout = Dropout(dropout)

		self.i_word_emb = Embedding(i_tokens.num(), d_emb)
		if share_word_emb: 
			assert i_tokens.num() == o_tokens.num()
			self.o_word_emb = self.i_word_emb
		else: self.o_word_emb = Embedding(o_tokens.num(), d_emb)

		self.encoder = SelfAttention(d_model, d_inner_hid, n_head, layers, dropout)
		self.decoder = Decoder(d_model, d_inner_hid, n_head, layers, dropout)
		self.target_layer = TimeDistributed(Dense(o_tokens.num(), use_bias=False))

	def compile(self, optimizer='adam', active_layers=999):
		src_seq_input = Input(shape=(None,), dtype='int32')
		tgt_seq_input = Input(shape=(None,), dtype='int32')

		src_seq = src_seq_input
		tgt_seq  = Lambda(lambda x:x[:,:-1])(tgt_seq_input)
		tgt_true = Lambda(lambda x:x[:,1:])(tgt_seq_input)

		src_emb = self.i_word_emb(src_seq)
		tgt_emb = self.o_word_emb(tgt_seq)

		if self.pos_emb: 
			src_emb = add_layer([src_emb, self.pos_emb(src_seq)])
			tgt_emb = add_layer([tgt_emb, self.pos_emb(tgt_seq)])
		src_emb = self.emb_dropout(src_emb)
		
		print('4444444444444444444444444')
		enc_output = self.encoder(src_emb, src_seq, active_layers=active_layers)
		print('5555555555555555555555555')
		dec_output = self.decoder(tgt_emb, tgt_seq, src_seq, enc_output, active_layers=active_layers)	
		final_output = self.target_layer(dec_output)

		def get_loss(y_pred, y_true):
			# add by lidongxing
			y_true = tf.cast(y_true, 'int32')
			y_ = self.label_smoothing(K.one_hot(y_true, self.o_tokens.num()))
			ce = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_)
			nonpadding = tf.cast(tf.not_equal(y_true, 0),'float32')  # 0: <pad>
			loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
			loss = K.mean(loss)

			return loss

		def get_accu(y_pred, y_true):
			mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
			corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
			corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
			return K.mean(corr)
				
		loss = get_loss(final_output, tgt_true)
		self.ppl = K.exp(loss)
		self.accu = get_accu(final_output, tgt_true)
		
		self.model = Model([src_seq_input, tgt_seq_input], final_output)
		# self.model = multi_gpu_model(self.model_, gpus=2)
		self.model.add_loss([loss])

		self.model.compile(optimizer, None)
		self.model.metrics_names.append('ppl')
		self.model.metrics.append(self.ppl)
		self.model.metrics_names.append('accu')
		self.model.metrics.append(self.accu)

	def make_src_seq_matrix(self, input_seqs):
		if type(input_seqs[0]) == type(''): input_seqs = [input_seqs]
		maxlen = max(map(len, input_seqs))
		src_seq = np.zeros((len(input_seqs), maxlen+3), dtype='int32')
		src_seq[:,0] = self.i_tokens.startid()
		for i, seq in enumerate(input_seqs):
			for ii, z in enumerate(seq):
				src_seq[i,1+ii] = self.i_tokens.id(z)
			src_seq[i,1+len(seq)] = self.i_tokens.endid()
		return src_seq

		
	def decode_sequence_readout_x(self, X, batch_size=32, max_output_len=64):
		if self.readout_model is None: self.make_readout_decode_model(max_output_len)
		target_seq = np.zeros((X.shape[0], 1), dtype='int32')
		target_seq[:,0] = self.o_tokens.startid()
		ret = self.readout_model.predict([X, target_seq], batch_size=batch_size, verbose=1)
		return ret

	def generate_sentence(self, rets, delimiter=''):
		sents = []
		for x in rets:
			end_pos = min([i for i, z in enumerate(x) if z == self.o_tokens.endid()]+[len(x)])
			rsent = [*map(self.o_tokens.token, x)][:end_pos]
			sents.append(delimiter.join(rsent))
		return sents

	def decode_sequence_readout(self, input_seqs, delimiter=''):
		if self.readout_model is None: self.make_readout_decode_model()
		src_seq = self.make_src_seq_matrix(input_seqs)
		target_seq = np.zeros((src_seq.shape[0],1), dtype='int32')
		target_seq[:,0] = self.o_tokens.startid()
		rets = self.readout_model.predict([src_seq, target_seq])
		rets = self.generate_sentence(rets, delimiter)
		if type(input_seqs[0]) is type('') and len(rets) == 1: rets = rets[0]
		return rets

	def make_fast_decode_model(self):
		src_seq_input = Input(shape=(None,), dtype='int32')
		src_emb = self.i_word_emb(src_seq_input)
		if self.pos_emb: src_emb = add_layer([src_emb, self.pos_emb(src_seq_input)])
		src_emb = self.emb_dropout(src_emb)
		enc_output = self.encoder(src_emb, src_seq_input)
		self.encode_model = Model(src_seq_input, enc_output)

		self.decoder_pre_step = DecoderPerStep(self.decoder)
		
		src_seq_input = Input(shape=(None,), dtype='int32')
		tgt_one_input = Input(shape=(1,), dtype='int32')
		enc_ret_input = Input(shape=(None, self.d_model))
		dec_ret_inputs = [Input(shape=(None, self.d_model)) for _ in self.decoder.layers]

		tgt_pos = Lambda(lambda x:tf.shape(x)[1])(dec_ret_inputs[0])

		tgt_one = self.o_word_emb(tgt_one_input)
		if self.pos_emb: tgt_one = add_layer([tgt_one, self.pos_emb(tgt_pos, pos_input=True)])

		dec_outputs = self.decoder_pre_step([tgt_one, src_seq_input, enc_ret_input]+dec_ret_inputs)	
		final_output = self.target_layer(dec_outputs[-1])

		self.decode_model = Model([tgt_one_input, src_seq_input, enc_ret_input]+dec_ret_inputs, 
							dec_outputs[:-1]+[final_output])
		

	def decode_sequence_fast(self, input_seqs, batch_size=32, delimiter='', verbose=0):
		if self.decode_model is None: self.make_fast_decode_model()
		src_seq = self.make_src_seq_matrix(input_seqs)

		start_mark, end_mark = self.o_tokens.startid(), self.o_tokens.endid()
		max_len = self.len_limit
		encode_model = self.encode_model
		decode_model = self.decode_model

		decode_batch = lambda x: decode_batch_greedy(x, encode_model, decode_model, start_mark, end_mark, max_len)
		
		rets = []
		rng = range(0, src_seq.shape[0], batch_size)
		if verbose and src_seq.shape[0] > batch_size: rng = tqdm(rng, total=len(rng))
		for iter in rng:
			rets.extend( decode_batch(src_seq[iter:iter+batch_size]) )
			
		rets = [delimiter.join(list(map(self.o_tokens.token, ret))) for ret in rets]
		if type(input_seqs[0]) is type('') and len(rets) == 1: rets = rets[0]
		return rets

	def beam_search(self, input_seqs, topk=4, batch_size=8 , length_penalty=1, delimiter='', verbose=0):
		if self.decode_model is None: self.make_fast_decode_model()
		src_seq = self.make_src_seq_matrix(input_seqs)

		start_mark, end_mark = self.o_tokens.startid(), self.o_tokens.endid()
		max_len = self.len_limit
		encode_model = self.encode_model
		decode_model = self.decode_model

		decode_batch = lambda x: decode_batch_beam_search(x, topk, encode_model, decode_model,
													start_mark, end_mark, max_len)
		
		rets = {}
		rng = range(0, src_seq.shape[0], batch_size)
		if verbose and src_seq.shape[0] > batch_size: rng = tqdm(rng, total=len(rng))

		for iter in rng:
			for i, x, y in decode_batch(src_seq[iter:iter+batch_size]):
				rets.setdefault(iter+i, []).append( (x, y/np.power(len(x)+1, length_penalty)) )
		rets = {x:sorted(ys,key=lambda x:x[-1], reverse=True) for x,ys in rets.items()}
		rets = [rets[i] for i in range(len(rets))]

		#
		rets = [[(delimiter.join(list(map(self.o_tokens.token, x))), y) for x, y in r] for r in rets]

		return rets
	
class PosEncodingLayer:
	def __init__(self, max_len, d_emb):
		self.pos_emb_matrix = Embedding(max_len, d_emb, trainable=False, \
						   weights=[GetPosEncodingMatrix(max_len, d_emb)])
	def get_pos_seq(self, x):
		mask = K.cast(K.not_equal(x, 0), 'int32')
		pos = K.cumsum(K.ones_like(x, 'int32'), 1)
		return pos * mask
	def __call__(self, seq, pos_input=False):
		x = seq
		if not pos_input: x = Lambda(self.get_pos_seq)(x)
		return self.pos_emb_matrix(x)

class AddPosEncoding:
	def __call__(self, x):
		_, max_len, d_emb = K.int_shape(x)
		pos = GetPosEncodingMatrix(max_len, d_emb)
		x = Lambda(lambda x:x+pos)(x)
		return x

class LRSchedulerPerStep(Callback):
	def __init__(self, d_model, warmup=4000):
		self.basic = d_model**-0.5
		self.warm = warmup**-1.5
		self.step_num = 0
	def on_batch_begin(self, batch, logs = None):
		self.step_num += 1
		lr = self.basic * min(self.step_num**-0.5, self.step_num*self.warm)
		K.set_value(self.model.optimizer.lr, lr)

add_layer = Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0])
# use this because keras may get wrong shapes with Add()([])
class LabelSmoothing:
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, inputs):
        V = inputs.get_shape().as_list()[-1]
        return ((1 - self.eps) * inputs) + (self.eps / V)

class TalkingHeadsAttention():

	def __init__(self, n_head, d_model, dropout=0.1, **kwargs):
		super(TalkingHeadsAttention, self).__init__(**kwargs)
		self.n_head = n_head
		self.d_k = self.d_v = d_k = d_v = d_model // n_head

		self.qs_layer = Dense(n_head * d_k, use_bias=False)
		self.ks_layer = Dense(n_head * d_k, use_bias=False)
		self.vs_layer = Dense(n_head * d_v, use_bias=False)

		self.ip1_dense = Dense(self.n_head * self.n_head, use_bias=False)
		self.ip2_dense = Dense(self.n_head, use_bias=False)

		self.dropout = Dropout(dropout)
		self.w_o = TimeDistributed(Dense(d_model))
	def __call__(self, q, k, v, mask=None):

		'''
		:param q: [B,lq,dmodel]
		:param k: [B,lk,dmodel]
		:param v: [B,lk,dmodel
		:param mask: [B,lq]
		:return:
		'''

		qs = self.qs_layer(q)			#[B,lq,dmodel]=[B,lq,heads*dk]
		ks = self.ks_layer(k)			#[B,lk,dmodel]=[B,lk,heads*dk]
		vs = self.vs_layer(v)			#[B,lk,dmodel]=[B,lk,heads*dv]

		def reshape1(x):
			s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
			x = tf.reshape(x, [s[0], s[1], self.n_head, s[2] // self.n_head])	# [batch_size, len_q, n_head, d_k]
			x = tf.transpose(x, [2, 0, 1, 3])	## [heads,batch_size, len_q,d_k]
			x = tf.reshape(x, [-1, s[1], s[2] // self.n_head])  # [n_head * batch_size, len_q, d_k]
			return x

		qs = Lambda(reshape1)(qs)		# [n_head * batch_size, len_q, d_k]
		ks = Lambda(reshape1)(ks)		# [n_head * batch_size, len_k, d_k]
		vs = Lambda(reshape1)(vs)		# [n_head * batch_size, len_k, d_v]

		if mask is not None:
			mask = Lambda(lambda x: K.repeat_elements(x, self.n_head, 0))(mask)		#[batch_size*heads,lq]
		temper = tf.sqrt(tf.cast(tf.shape(ks)[-1], dtype='float32'))		#d_k
		attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)([qs, ks])  # shape=(batch*heads, lq, lk)
		if mask is not None:
			mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(mask)
			attn = Add()([attn, mmask])		# shape=(batch*heads, lq, lk)

		def reshape2(x):
			s = tf.shape(x)
			x = tf.reshape(x,[s[0]//self.n_head,self.n_head,s[1],s[2]])
			x = tf.transpose(x,[0,2,3,1])
			return x

		attn = Lambda(reshape2)(attn)		# shape=(batch_size, lq, lk,heads)
		attn = self.ip1_dense(attn)		#[B,lq,lk,heads*heads]
		attn = self.ip2_dense(attn)		#[B,lq,lk,heads]

		def reshape3(x):
			s = tf.shape(x)
			x = tf.transpose(x,[0,3,1,2])
			x = tf.reshape(x,[-1,s[1],s[2]])
			return x

		attn = Lambda(reshape3)(attn)
		attn = Activation('softmax')(attn)
		attn = self.dropout(attn)
		output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, vs])		# shape=(batch*heads, lq, dv)

		def reshape4(x):
			s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
			x = tf.reshape(x, [self.n_head, -1, s[1], s[2]])
			x = tf.transpose(x, [1, 2, 0, 3])
			x = tf.reshape(x, [-1, s[1], self.n_head*self.d_v])	# [batch_size, len_v, n_head * d_v]
			return x

		output = Lambda(reshape4)(output)	# [batch_size, len_v, n_head * d_v]

		output = self.w_o(output)		# [batch_size, len_v, dmodel]
		output = self.dropout(output)		# [batch_size, len_v, dmodel]
		return output, attn

class TalkingHeadsAttention2():

    def __init__(self, n_head, d_model, dropout=0.1, **kwargs):
        super(TalkingHeadsAttention, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head

        self.qs_layer = Dense(n_head * d_k, use_bias=False)
        self.ks_layer = Dense(n_head * d_k, use_bias=False)
        self.vs_layer = Dense(n_head * d_v, use_bias=False)

        self.ip1_dense = Dense(self.n_head * self.n_head, use_bias=False)
        self.ip2_dense = Dense(self.n_head, use_bias=False)

        self.dropout = Dropout(dropout)
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):

        '''
        :param q: [B,lq,dmodel]
        :param k: [B,lk,dmodel]
        :param v: [B,lk,dmodel
        :param mask: [B,lq]
        :return:
        '''

        qs = self.qs_layer(q)  # [B,lq,dmodel]=[B,lq,heads*dk]
        ks = self.ks_layer(k)  # [B,lk,dmodel]=[B,lk,heads*dk]
        vs = self.vs_layer(v)  # [B,lk,dmodel]=[B,lk,heads*dv]

        def reshape1(x):
            s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
            x = tf.reshape(x, [s[0], s[1], self.n_head, s[2] // self.n_head])  # [batch_size, len_q, n_head, d_k]
            x = tf.transpose(x, [2, 0, 1, 3])  ## [heads,batch_size, len_q,d_k]
            x = tf.reshape(x, [-1, s[1], s[2] // self.n_head])  # [n_head * batch_size, len_q, d_k]
            return x

        qs = Lambda(reshape1)(qs)  # [n_head * batch_size, len_q, d_k]
        ks = Lambda(reshape1)(ks)  # [n_head * batch_size, len_k, d_k]
        vs = Lambda(reshape1)(vs)  # [n_head * batch_size, len_k, d_v]

        if mask is not None:
            mask = Lambda(lambda x: K.repeat_elements(x, self.n_head, 0))(mask)  # [batch_size*heads,lq]
        temper = tf.sqrt(tf.cast(tf.shape(ks)[-1], dtype='float32'))  # d_k
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)([qs, ks])  # shape=(batch*heads, lq, lk)
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(mask)
            attn = Add()([attn, mmask])  # shape=(batch*heads, lq, lk)


        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('@@@@@@@@@@pretalking heads@@@@@@@')
        print('@@@@@@@@@@pretalking heads@@@@@@@')

        def reshape2(x):
            s = tf.shape(x)
            x = tf.reshape(x, [s[0] // self.n_head, self.n_head, s[1], s[2]])
            x = tf.transpose(x, [0, 2, 3, 1])
            return x

        def reshape3(x):
            s = tf.shape(x)
            x = tf.transpose(x, [0, 3, 1, 2])
            x = tf.reshape(x, [-1, s[1], s[2]])
            return x

        attn = Lambda(reshape2)(attn)  # shape=(batch_size, lq, lk,heads)
        attn = self.ip1_dense(attn)  # [B,lq,lk,heads*heads]
        attn = self.ip2_dense(attn)  # [B,lq,lk,heads]
        attn = Lambda(reshape3)(attn)       # shape=(batch*heads, lq, lk)
        attn = Activation('softmax')(attn)      # shape=(batch*heads, lq, lk)


        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('@@@@@@@@posttalking heads@@@@@@@@@@@')
        print('@@@@@@@@posttalking heads@@@@@@@@@@@')

        attn = Lambda(reshape2)(attn)       # shape=(batch_size, lq, lk,heads)
        attn = self.ip1_dense(attn)  # [B,lq,lk,heads*heads]
        attn = self.ip2_dense(attn)  # [B,lq,lk,heads]
        attn = Lambda(reshape3)(attn)  # shape=(batch*heads, lq, lk)
        attn = Activation('softmax')(attn)  # shape=(batch*heads, lq, lk)


        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, vs])  # shape=(batch*heads, lq, dv)

        def reshape4(x):
            s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
            x = tf.reshape(x, [self.n_head, -1, s[1], s[2]])
            x = tf.transpose(x, [1, 2, 0, 3])
            x = tf.reshape(x, [-1, s[1], self.n_head * self.d_v])  # [batch_size, len_v, n_head * d_v]
            return x

        output = Lambda(reshape4)(output)  # [batch_size, len_v, n_head * d_v]

        output = self.w_o(output)  # [batch_size, len_v, dmodel]
        output = self.dropout(output)  # [batch_size, len_v, dmodel]
        return output, attn

class InteractingHeads():
	def __init__(self, heads,d_model, dropout=0.1):
		self.heads = heads
		print('000000000000000')
		print(self.heads)

		self.d_k = self.d_v = d_k = d_v = d_model // heads

		self.qs_layer = Dense(heads * d_k, use_bias=False)
		self.ks_layer = Dense(heads * d_k, use_bias=False)
		self.vs_layer = Dense(heads * d_v, use_bias=False)

		self.dropout = Dropout(dropout)
		self.w_o = TimeDistributed(Dense(d_model))

	def __call__(self,q, k, v, mask=None):
		'''
		:param q: [batch_size,lq,dmodel]
		:param k: [batch_size,lk,dmodel]
		:param v: [batch_size,lk,dmodel]
		:param mask: encoder_self_mask:[batch_size,lq]  decoder_self_mask:[batch_size,lq,lq]
					decoder_encoder_mask:[batch_size,lq,lk]
		:return:[batch_size,lq,dmodel]
		'''

		qs = self.qs_layer(q)  # [B,lq,dmodel]=[B,lq,heads*dk]
		ks = self.ks_layer(k)  # [B,lk,dmodel]=[B,lk,heads*dk]
		vs = self.vs_layer(v)  # [B,lk,dmodel]=[B,lk,heads*dv]

		def reshape1(x):
			s = tf.shape(x)
			x = tf.reshape(x, [s[0], s[1], self.heads, s[2] // self.heads])
			x = tf.transpose(x, [0, 2, 1, 3])
			return x

		qs = Lambda(reshape1)(qs)  # [batch,head, lq,dk]
		ks = Lambda(reshape1)(ks)  # [batch,head, lk,dk]
		vs = Lambda(reshape1)(vs)  # [batch,head, lk,dv]



		temper = tf.sqrt(tf.cast(tf.shape(ks)[-1], dtype='float32')) #[dk]



		attn = []  # heads*heads attention
		for i in range(self.heads):
			for j in range(self.heads):
				a_ij = Lambda(lambda x: K.batch_dot(x[0][:, i, :, :], x[1][:, j, :, :],axes=[2,2]) / temper)([qs, ks])  # [batch_size,len_q,len_k]
				if mask is not None:
					mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(mask)  # [batch_size,lq]
					a_ij = Add()([a_ij, mmask])		# [batch_size,qlen,klen]
				# a_ij = Activation('softmax')(a_ij)
				# a_ij = self.dropout(a_ij)
				a_ij = Lambda(lambda x: K.expand_dims(x, axis=1))(a_ij)  # [batch_size,1,qlen,klen]
				attn.append(a_ij)
		outputs = []
		attns = []
		j = 0
		while True:
			if (j != 0) and (j % self.heads == 0):
				break
			temp_a = []
			for i in range(self.heads):
				a = attn[i * self.heads + j]  # [batch_size,1,qlen,klen]
				temp_a.append(a)
			sm = Add()([x for x in temp_a]) #[batch_size,1,qlen,klen]
			sm = Activation('softmax')(sm)
			sm = self.dropout(sm)
			output = Lambda(lambda x: K.batch_dot(x[0][:, 0, :, :], x[1][:, j, :, :],axes=[2,1]))([sm, vs])  # [batch_size,qlen,dv]

			output = Lambda(lambda x: K.expand_dims(x, axis=0))(output) #[1,batch_size,qlen,dv]
			sm = Lambda(lambda x: K.permute_dimensions(x,(1,0,2,3)))(sm)
			outputs.append(output)
			attns.append(sm)
			j = j + 1

		# len(attns)=heads, attns[0]# [1,batch_size,qlen,klen]
		attns = Lambda(lambda x:K.concatenate(x,axis=0))([x for x in attns])		#[heads,batch_size,qlen,klen]

		def reshape2(x):
			s = tf.shape(x)
			x = tf.reshape(x,[-1,s[2],s[3]])
			return x
		attns = Lambda(reshape2)(attns)				#[heads*batch_size,qlen,klen]

		outputs = Lambda(lambda x: K.concatenate(x, axis=0))([x for x in outputs])  # [heads,batch_size,qlen,dv]

		def reshape3(x):
			s = tf.shape(x)			# [heads,batch_size,qlen,dv]
			x = tf.transpose(x,[1,2,0,3])		# [batch_size,qlen,dv,heads]
			x = tf.reshape(x, [s[1], s[2], self.d_v*self.heads])  # [batch, qlen,dmodel]
			return x

		ret = Lambda(reshape3)(outputs)		# [batch, qlen,dmodel]
		ret = self.dropout(ret) # [batch, qlen,dmodel]
		return ret, attns			# attn:shape=(batch*heads, lq, lk)  ret:[batch_size, len_v, dmodel]


if __name__ == '__main__':
	x = InteractingHeads(8,256)
	print('done')
