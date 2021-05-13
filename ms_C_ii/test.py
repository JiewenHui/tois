import sys
import os
import subprocess
import time
batch_size = [3]
num_filters=[80]
l2_reg_lambda=[0]
learning_rate = [0.01]
dropout_keep_probs = [0.1]
count = 0
for batch in batch_size:
	for num in num_filters:
		for d in dropout_keep_probs:
				for l2 in l2_reg_lambda:
					for rate in learning_rate:
						print ('The ', count, 'excue\n')
						count += 1
						if learning_rate== 0.0001:
							subprocess.call('python train.py --batch_size %d --num_filters %d --l2_reg_lambda %f --learning_rate %f --num_epochs %d --dropout_keep_probs %f --filter_sizes %d' % (batch,num,l2,rate,50,d,40), shell = True)
						else:
							subprocess.call('python train.py --batch_size %d --num_filters %d --l2_reg_lambda %f --learning_rate %f --num_epochs %d --dropout_keep_probs %f --filter_sizes %d' % (batch,num,l2,rate,50,d,40), shell = True)
