#!/usr/bin/env python
#encoding=utf-8

import tensorflow as tf
import numpy as np
# point_wise obbject
rng = np.random.RandomState(23455)
class NNQLM_I(object):
    def __init__(self, max_input_left, max_input_right, vocab_size,embedding_size,batch_size,
      embeddings,dropout_keep_prob,filter_sizes, 
      num_filters,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable = True,overlap_needed = True,position_needed = True,pooling = 'max',hidden_num = 10,\
      extend_feature_dim = 10):
        self.dropout_keep_prob = dropout_keep_prob
        self.num_filters = num_filters
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.overlap_needed = overlap_needed
        self.vocab_size = vocab_size
        self.trainable = trainable
        self.filter_sizes = filter_sizes
        self.pooling = pooling
        self.position_needed = position_needed
        if self.overlap_needed:
            self.total_embedding_dim = embedding_size + extend_feature_dim
        else:
            self.total_embedding_dim = embedding_size
        if self.position_needed:
            self.total_embedding_dim = self.total_embedding_dim + extend_feature_dim
        self.batch_size = batch_size
        self.l2_reg_lambda = l2_reg_lambda
        self.para = []
        self.max_input_left = max_input_left
        self.max_input_right = max_input_right
        self.hidden_num = hidden_num
        self.extend_feature_dim = extend_feature_dim
        self.is_Embedding_Needed = is_Embedding_Needed
        self.rng = 23455
    def create_placeholder(self):
        self.question = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[self.batch_size,self.max_input_right],name = 'input_answer')
        self.input_y = tf.placeholder(tf.float32, [self.batch_size,2], name = "input_y")
        self.q_overlap = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'q_feature_embeding')
        self.a_overlap = tf.placeholder(tf.int32,[self.batch_size,self.max_input_right],name = 'a_feature_embeding')
        self.q_position = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'q_position')
        self.a_position = tf.placeholder(tf.int32,[self.batch_size,self.max_input_right],name = 'a_position')
    def density_weighted(self):
        self.weighted_q = tf.Variable(tf.ones([1,self.max_input_left,1,1]) , name = 'weighted_q')
        self.para.append(self.weighted_q)
        self.weighted_a = tf.Variable(tf.ones([1,self.max_input_right,1,1]) , name = 'weighted_a')
        self.para.append(self.weighted_a)
    def add_embeddings(self):

        # Embedding layer for both CNN
        with tf.name_scope("embedding"):
            if self.is_Embedding_Needed:
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = self.trainable )
            else:
                W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)
            self.embedding_W = W
            self.overlap_W =  tf.get_variable('overlap_w', shape=[3, self.embedding_size],initializer = tf.random_normal_initializer())#tf.Variable(tf.random_uniform([3, self.extend_feature_dim], -1.0, 1.0),name="W",trainable = True)
            self.position_W = tf.Variable(tf.random_uniform([300,self.embedding_size], -1.0, 1.0),name = 'W',trainable = True)
            # self.para.append(self.embedding_W)
            # self.para.append(self.overlap_W)

        #get embedding from the word indices
        self.embedded_chars_q = self.concat_embedding(self.question,self.q_overlap,self.q_position)

        self.embedded_chars_a = self.concat_embedding(self.answer,self.a_overlap,self.a_position)
    def joint_representation(self):
        self.density_q = self.density_matrix(self.embedded_chars_q,self.weighted_q)
        self.density_a = self.density_matrix(self.embedded_chars_a,self.weighted_a)
        self.M_qa = tf.matmul(self.density_q,self.density_a)
    def direct_representation(self):

        self.embedded_q = tf.reshape(self.embedded_chars_q,[-1,self.max_input_left,self.total_embedding_dim])
        self.embedded_a = tf.reshape(self.embedded_chars_a,[-1,self.max_input_right,self.total_embedding_dim])
        reverse_a = tf.transpose(self.embedded_a,[0,2,1])
        self.M_qa = tf.matmul(self.embedded_q,reverse_a)

    def trace_represent(self):
        self.density_diag = tf.matrix_diag_part(self.M_qa)
        self.density_trace = tf.expand_dims(tf.trace(self.M_qa),-1)
        self.match_represent = tf.concat([self.density_diag,self.density_trace],1)
        self.represent = self.match_represent
    #construct the density_matrix
    def density_matrix(self,sentence_matrix,sentence_weighted):
        # print sentence_matrix
        # print tf.nn.l2_normalize(sentence_matrix,2)

        self.norm = tf.nn.l2_normalize(sentence_matrix,2)
        # self.norm = tf.nn.softmax(sentence_matrix,2)
        # self.norm = sentence_matrix
        # print tf.reduce_sum(norm,2)
        reverse_matrix = tf.transpose(self.norm, perm = [0,1,3,2])
        q_a = tf.matmul(self.norm,reverse_matrix)
        # return tf.reduce_sum(tf.matmul(self.norm,reverse_matrix), 1)
        return tf.reduce_sum(tf.multiply(q_a,sentence_weighted),1)


    def feed_neural_work(self):     
        with tf.name_scope('regression'):
            #W = tf.Variable(tf.zeros(shape = [(self.total_embedding_dim - self.filter_sizes[0] + 1) * self.num_filters * 2,2]),name = 'W') 
            W = tf.Variable(tf.zeros(shape = [self.embedding_size + 1,2]),name = 'W') 
        # with tf.name_scope('neural_network'):
        #     W = tf.get_variable(
        #         "W_hidden",
        #         shape=[(self.total_embedding_dim - self.filter_sizes[0] + 1) * self.num_filters * 2,self.hidden_num],
        #         # shape = [self.total_embedding_dim + 1,self.hidden_num],
        #         initializer = tf.contrib.layers.xavier_initializer())
        #     b = tf.get_variable('b_hidden', shape=[self.hidden_num],initializer = tf.random_normal_initializer())
        #     self.para.append(W)
        #     self.para.append(b)
        #     self.hidden_output = tf.nn.tanh(tf.nn.xw_plus_b(self.represent, W, b, name = "hidden_output"))
        # #add dropout
        # with tf.name_scope('dropout'):
        #     self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
        # with tf.name_scope("output"):
        #     W = tf.get_variable(
        #         "W_output",
        #         shape = [self.hidden_num, 2],
        #         initializer = tf.contrib.layers.xavier_initializer())
        #     b = tf.get_variable('b_output', shape=[2],initializer = tf.random_normal_initializer())
            b = tf.Variable(tf.zeros([2]),name = 'b')
            self.para.append(W)
            self.para.append(b)
            self.logits = tf.nn.xw_plus_b(self.represent, W, b, name = "scores")
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
            
    def create_loss(self):
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y)
            #pi_regularization = tf.reduce_sum(self.weighted_q) - 1 + tf.reduce_sum(self.weighted_a) - 1
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def concat_embedding(self,words_indice,overlap_indice,position_indice):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        position_embedding = tf.nn.embedding_lookup(self.position_W,position_indice)
        overlap_embedding_q = tf.nn.embedding_lookup(self.overlap_W,overlap_indice)
        if not self.overlap_needed :
            if not self.position_needed:
                return tf.expand_dims(embedded_chars_q,-1)
            else:
                return tf.expand_dims(tf.reduce_sum([embedded_chars_q,position_embedding],0),-1)
        else:
            if not self.position_needed:
                return  tf.expand_dims(tf.reduce_sum([embedded_chars_q,overlap_embedding_q],0),-1)
            else:
                return tf.expand_dims(tf.reduce_sum([embedded_chars_q,overlap_embedding_q,position_embedding],0),-1)
        

    def build_graph(self):
        self.create_placeholder()
        self.add_embeddings()
        self.density_weighted()
        self.joint_representation()
        # self.direct_representation()
        self.trace_represent()
        # self.convolution()
        # self.pooling_graph()
        # self.interact()
        self.feed_neural_work()
        self.create_loss()

if __name__ == '__main__':
    cnn = QA_quantum(max_input_left = 33,
                max_input_right = 40,
                vocab_size = 5000,
                embedding_size = 50,
                batch_size = 3,
                embeddings = None,
                dropout_keep_prob = 1,
                filter_sizes = [40],
                num_filters = 65,
                l2_reg_lambda = 0.0,
                is_Embedding_Needed = False,
                trainable = True,
                overlap_needed = False,
                pooling = 'max',
                position_needed = False)
    cnn.build_graph()
    input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_y = np.ones((3,2))

    input_overlap_q = np.ones((3,33))
    input_overlap_a = np.ones((3,40))
    q_posi = np.ones((3,33))
    a_posi = np.ones((3,40))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            cnn.input_y:input_y,
            cnn.q_overlap:input_overlap_q,
            cnn.a_overlap:input_overlap_a,
            cnn.q_position:q_posi,
            cnn.a_position:a_posi
        }
       
        see,question,answer,scores = sess.run([cnn.embedded_chars_q,cnn.question,cnn.answer,cnn.scores],feed_dict)
        # print see

       
