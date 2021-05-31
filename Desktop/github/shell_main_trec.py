import sys
import os
import subprocess
import time

batch_size = [128,64,32]
dropout_keep_probs=[0.8]
learning_rate = [0.0001]
num_filters = [80]
num_epoch = 10
l2_reg_lambda = [0.0001]
count = 0
for batch in batch_size:
	for dropout_keep_prob in dropout_keep_probs:
		for rate in learning_rate:
			for l2 in l2_reg_lambda:
				for num_filter in num_filters:
					print ('The ', count, 'excue\n')
					count += 1
					subprocess.call('python train.py --batch_size %d --dropout_keep_prob %f --learning_rate %f --num_epochs %d --num_filters %d  --l2_reg_lambda %d' % (batch,dropout_keep_prob,rate,num_epoch,num_filter,l2), shell = True)
