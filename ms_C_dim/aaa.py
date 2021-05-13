import tensorflow as tf
# label = [0]*5
# label[4],label[2] = 1,1
# print(label)
# gt = []
# for i,v in enumerate(label):
# 	if v>0:
# 		gt.append(i)
# print(set(gt))
sess = tf.Session()
label = [0,1,0,0,0,0,1,1]
xx = tf.range(0,8) * 2 + label
print(sess.run(xx))