import  tensorflow as tf
from  bert import modeling
import tensorflow as tf
import numpy as np
from multiply import ComplexMultiply
import math
from scipy import linalg
# point_wise obbject
from numpy.random import RandomState
rng = np.random.RandomState(23455)
# from complexnn.dense import ComplexDense
# from complexnn.utils import GetReal
import math

class TextConfig():

    seq_length=64        #max length of sentence
    num_labels=2         #number of labels

    num_filters=128        #number of convolution kernel
    filter_sizes=[2,3,4]   #size of convolution kernel
    hidden_dim=128         #number of fully_connected layer units

    keep_prob=0.5          #droppout
    lr= 1e-6               #learning rate
    lr_decay= 0.9          #learning rate decay
    clip= 5.0              #gradient clipping threshold

    is_training=True       #is _training
    use_one_hot_embeddings=False  #use_one_hot_embeddings


    num_epochs=50          #epochs
    batch_size=1       #batch_size
    print_per_batch =100   #print result
    require_improvement=2000   #stop training if no inporement over 1000 global_step

    output_dir='./result'
    # data_dir='../data/corpus/cnews'  #the path of input_data file
    # vocab_file = '../chinese_L-12_H-768_A-12/vocab.txt'  #the path of vocab file
    # bert_config_file='../chinese_L-12_H-768_A-12/bert_config.json'  #the path of bert_cofig file
    # init_checkpoint ='../chinese_L-12_H-768_A-12/bert_model.ckpt'   #the path of bert model
    # data_dir='../data/wiki'  #the path of input_data file
    data_dir='/root/hwj/new_data/ms_small'  #the path of input_data file
    # vocab_file = '../uncased_L-2_H-128_A-2/vocab.txt'  #the path of vocab file
    # bert_config_file='../uncased_L-2_H-128_A-2/bert_config.json'  #the path of bert_cofig file
    # init_checkpoint ='../uncased_L-2_H-128_A-2/bert_model.ckpt'   #the path of bert model
    # data_dir='../data/wiki'  #the path of input_data file
    vocab_file = '/root/hwj/bert/uncased_L-24_H-1024_A-16/vocab.txt'  #the path of vocab file
    bert_config_file='/root/hwj/bert/uncased_L-24_H-1024_A-16/bert_config.json'  #the path of bert_cofig file
    init_checkpoint ='/root/hwj/bert/uncased_L-24_H-1024_A-16/bert_model.ckpt'   #the path of bert model

class TextCNN(object):

    def __init__(self,config):
        '''获取超参数以及模型需要的传入的5个变量，input_ids，input_mask，segment_ids，labels，keep_prob'''
        self.config=config
        self.bert_config = modeling.BertConfig.from_json_file(self.config.bert_config_file)

        self.input_ids=tf.placeholder(tf.int64,shape=[None,self.config.seq_length],name='input_ids')
        self.input_mask=tf.placeholder(tf.int64,shape=[None,self.config.seq_length],name='input_mask')
        self.segment_ids=tf.placeholder(tf.int64,shape=[None,self.config.seq_length],name='segment_ids')
        self.labels=tf.placeholder(tf.int64,shape=[None,],name='labels')
        self.keep_prob=tf.placeholder(tf.float32,name='dropout')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.q_length = tf.placeholder(tf.int32,[config.batch_size,1],name = 'q_length')
        self.a_length = tf.placeholder(tf.int32,[config.batch_size,1],name = 'a_length')

        self.cnn()
    def cnn(self):

        '''获取bert模型最后的token-level形式的输出(get_sequence_output)，将此作为embedding_inputs，作为卷积的输入'''
        with tf.name_scope('bert'):
            bert_model = modeling.BertModel(
                config=self.bert_config,
                is_training=self.config.is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=self.config.use_one_hot_embeddings)
            embedding_inputs= bert_model.get_sequence_output()

            # print(self.a)
            # exit()
        '''用三个不同的卷积核进行卷积和池化，最后将三个结果concat'''
        with tf.name_scope('conv'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size,reuse=False):
                    conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, filter_size,name='conv1d')
                    pooled = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                    pooled_outputs.append(pooled)

            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 1)
            outputs = tf.reshape(h_pool, [-1, num_filters_total])

        '''加全连接层和dropuout层'''
        with tf.name_scope('fc'):
            fc=tf.layers.dense(outputs,self.config.hidden_dim,name='fc1')
            fc = tf.nn.dropout(fc, self.keep_prob)
            fc=tf.nn.relu(fc)

        '''logits'''
        with tf.name_scope('logits'):
            self.logits = tf.layers.dense(fc, self.config.num_labels, name='logits')
            self.prob = tf.nn.softmax(self.logits)
            self.scores = self.prob
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        '''计算loss，因为输入的样本标签不是one_hot的形式，需要转换下'''
        with tf.name_scope('loss'):
            log_probs = tf.nn.log_softmax(self.logits, axis=-1)
            one_hot_labels = tf.one_hot(self.labels, depth=self.config.num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            self.loss = tf.reduce_mean(per_example_loss)

        '''optimizer'''
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        '''accuracy'''
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.labels, self.y_pred_cls)
            self.acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


class QA_quantum(object):
    def __init__(
      self, config,max_input_left, max_input_right, vocab_size,embedding_size,batch_size,
      dropout_keep_prob,filter_sizes, 
      num_filters,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable = True,overlap_needed = True,position_needed = True,pooling = 'max',hidden_num = 10,\
      extend_feature_dim = 10):

        
        self.dropout_keep_prob = dropout_keep_prob
        self.num_filters = num_filters
        # self.embeddings = embeddings
        # self.embeddings_complex=embeddings_complex
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

        self.config=config

    # def jiangwei(self):
    #     regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg_lambda) 
    #     W = tf.get_variable( "W_jiang",
    #             #shape=[102,self.hidden_num],
    #         shape=[768,100],
    #         initializer = tf.contrib.layers.xavier_initializer(),
    #         regularizer=regularizer)
    #     b = tf.get_variable('b_jiang', shape=[100],initializer = tf.random_normal_initializer(),regularizer=regularizer)
    #     self.embedding_inputs = tf.nn.tanh(tf.nn.xw_plus_b(self.embeddings, W, b, name = "jiang_output"))
    #     print('='*100)
    #     print('self.embeddings: ',self.embedding_inputs)
    #     print('='*100)
    def create_placeholder(self):
        # self.question = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'input_question')
        # self.answer = tf.placeholder(tf.int32,[self.batch_size,self.max_input_right],name = 'input_answer')
        # self.input_y = tf.placeholder(tf.float32, [self.batch_size,2], name = "input_y")
        # self.q_position = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'q_position')
        # self.a_position = tf.placeholder(tf.int32,[self.batch_size,self.max_input_right],name = 'a_position')
        # # self.overlap = tf.placeholder(tf.float32,[self.batch_size,2],name = 'a_position')
        # self.q_overlap = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'q_position')
        # self.a_overlap = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'a_position')

        #==================================================#
        #--------------------------------------------------#
        self.bert_config = modeling.BertConfig.from_json_file(self.config.bert_config_file)

        self.input_ids=tf.placeholder(tf.int64,shape=[None,self.config.seq_length],name='input_ids')
        self.input_mask=tf.placeholder(tf.int64,shape=[None,self.config.seq_length],name='input_mask')
        self.segment_ids=tf.placeholder(tf.int64,shape=[None,self.config.seq_length],name='segment_ids')
        self.labels=tf.placeholder(tf.int64,shape=[None,],name='labels')
        self.keep_prob=tf.placeholder(tf.float32,name='dropout')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        #--------------------------------------------------#
        # self.q_position = tf.placeholder(tf.int32,[self.batch_size,self.q_length[0]],name = 'q_position')
        # self.a_position = tf.placeholder(tf.int32,[self.batch_size,self.a_length[0]],name = 'a_position')
        self.q_length = tf.placeholder(tf.int32,[None,1],name = 'q_length')
        self.a_length = tf.placeholder(tf.int32,[None,1],name = 'a_length')
        #==================================================#
    def Embedding(self):
        self.embedding_W_pos = tf.Variable(self.Position_Embedding(self.embedding_size),name = 'W',trainable = True)
        with tf.name_scope('bert'):
            bert_model = modeling.BertModel(
                config=self.bert_config,
                is_training=self.config.is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=self.config.use_one_hot_embeddings)
            embedding_inputs= bert_model.get_sequence_output()
            temp = embedding_inputs.shape[1]
            embedding_inputs = tf.reshape(embedding_inputs,[-1,embedding_inputs.shape[-1]])

            regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg_lambda) 
            W = tf.get_variable( "W_jiang",
                #shape=[102,self.hidden_num],
                shape=[1024,self.embedding_size],
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer=regularizer)
            b = tf.get_variable('b_jiang', shape=[self.embedding_size],initializer = tf.random_normal_initializer(),regularizer=regularizer)
            embedding_inputs = tf.nn.tanh(tf.nn.xw_plus_b(embedding_inputs, W, b, name = "jiang_output"))
            embedding_inputs = tf.reshape(embedding_inputs,[-1,temp,self.embedding_size])
            print('='*100)
            print('self.embeddings: ',embedding_inputs)
            print('='*100)
           
            self.q_real,self.a_real = [],[]
            self.q_phase,self.a_phase = [],[]
            for i in range(0,self.batch_size):
                temp = self.q_length[i][0]
                q_real = embedding_inputs[i][:temp]
                a_real = embedding_inputs[i][temp:]

                

                q_words_indice = self.input_ids[i][:temp]
                q_phase=tf.nn.embedding_lookup(self.embedding_W_pos,q_words_indice)
                
                q_phase=tf.nn.dropout(q_phase, self.dropout_keep_prob, name="hidden_output_drop")
                
                
                a_words_indice = self.input_ids[i][temp:]
                a_phase=tf.nn.embedding_lookup(self.embedding_W_pos,a_words_indice)
                a_phase=tf.nn.dropout(a_phase, self.dropout_keep_prob, name="hidden_output_drop")


                q_real_trans = tf.transpose(q_real,perm=[1,0])
                q_phase_trans = tf.transpose(q_phase,perm=[1,0])
                a_real_trans = tf.transpose(a_real,perm=[1,0])
                a_phase_trans = tf.transpose(a_phase,perm=[1,0])

                q_real_real = tf.matmul(q_real_trans,q_real)
                q_imag_imag = tf.matmul(q_phase_trans,q_phase)
                a_real_real = tf.matmul(a_real_trans,a_real)
                a_imag_imag = tf.matmul(a_phase_trans,a_phase)

                self.q_real.append(q_real_real-q_imag_imag)
                self.q_phase.append(q_real_real+q_imag_imag)
                self.a_real.append(a_real_real-a_imag_imag)
                self.a_phase.append(a_real_real+a_imag_imag)
            self.q_real = tf.expand_dims(self.q_real,0)
            self.q_phase = tf.expand_dims(self.q_phase,0)
            self.a_real = tf.expand_dims(self.a_real,0)
            self.a_phase = tf.expand_dims(self.a_phase,0)

            self.q_real = tf.reshape(self.q_real,[self.batch_size,self.embedding_size,self.embedding_size])
            self.q_phase = tf.reshape(self.q_phase,[self.batch_size,self.embedding_size,self.embedding_size])
            self.a_real = tf.reshape(self.a_real,[self.batch_size,self.embedding_size,self.embedding_size])
            self.a_phase = tf.reshape(self.a_phase,[self.batch_size,self.embedding_size,self.embedding_size])

            self.M_qa_real=tf.matmul(self.q_real,self.a_real)+tf.matmul(self.q_phase,self.a_phase)
            self.M_qa_imag=tf.matmul(self.q_phase,self.a_real)-tf.matmul(self.q_real,self.a_phase)
   
    def Position_Embedding(self,position_size):
        batch_size=self.batch_size
        seq_len = self.vocab_size

        position_j = 1. / tf.pow(10000., 2 * tf.range(position_size, dtype=tf.float32) / position_size)
        position_j = tf.expand_dims(position_j, 0)

        position_i=tf.range(tf.cast(seq_len,tf.float32), dtype=tf.float32) + 1
        position_i=tf.expand_dims(position_i,1)

       
        position_ij = tf.matmul(position_i, position_j)
        position_embedding = position_ij

        return position_embedding

    
    def feed_neural_work(self):
        with tf.name_scope('regression'):
            regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
            W = tf.get_variable( "W_hidden",
                #shape=[102,self.hidden_num],
                shape=[self.represent.shape[-1],self.hidden_num],
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer=regularizer)
            b = tf.get_variable('b_hidden', shape=[self.hidden_num],initializer = tf.random_normal_initializer(),regularizer=regularizer)
            self.para.append(W)
            self.para.append(b)
            self.hidden_output = tf.nn.tanh(tf.nn.xw_plus_b(self.represent, W, b, name = "hidden_output"))
            #self.hidden_output=tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
            W = tf.get_variable(
                "W_output",
                shape = [self.hidden_num, 2],
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer=regularizer)
            b = tf.get_variable('b_output', shape=[2],initializer = tf.random_normal_initializer(),regularizer=regularizer)
            self.para.append(W)
            self.para.append(b)
            self.logits = tf.nn.xw_plus_b(self.hidden_output, W, b, name = "scores")
            print(self.logits)
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
    def create_loss(self):
        self.weighted_q = tf.Variable(tf.ones([1,self.max_input_left,1,1]) , name = 'weighted_q')
        self.weighted_q=tf.nn.softmax(self.weighted_q,1)
        self.weighted_a = tf.Variable(tf.ones([1,self.max_input_right,1,1]) , name = 'weighted_a')
        self.weighted_a=tf.nn.softmax(self.weighted_a,1)
        l2_loss = tf.constant(0.0)
        
        with tf.name_scope('loss'):
            log_probs = tf.nn.log_softmax(self.logits, axis=-1)
            one_hot_labels = tf.one_hot(self.labels, depth=self.config.num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            self.loss = tf.reduce_mean(per_example_loss)
        '''optimizer'''
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.labels)
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    
    def convolution(self):
        #initialize my conv kernel
        self.kernels_real = []
        self.kernels_imag=[]
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-pool-%s' % filter_size):
                filter_shape = [filter_size,filter_size,1,self.num_filters]
                input_dim=2
                fan_in = np.prod(filter_shape[:-1])
                fan_out = (filter_shape[-1] * np.prod(filter_shape[:2]))
                s=1./fan_in
                rng=RandomState(23455)
                modulus=rng.rayleigh(scale=s,size=filter_shape)
                phase=rng.uniform(low=-np.pi,high=np.pi,size=filter_shape)
                W_real=modulus*np.cos(phase)
                W_imag=modulus*np.sin(phase)
                W_real = tf.Variable(W_real,dtype = 'float32')
                W_imag = tf.Variable(W_imag,dtype = 'float32')
                self.kernels_real.append(W_real)
                self.kernels_imag.append(W_imag)
                # self.para.append(W_real)
                # self.para.append(W_imag)
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        # self.qa_real = self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)-self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)
        # print(self.qa_real)
        # self.qa_imag = self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)+self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)
        # print(self.qa_imag)
        # self.qa_real_0 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[0]+tf.expand_dims(self.M_qa_real,-1))-tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[0]+tf.expand_dims(self.M_qa_real,-1))
        # self.qa_real_1 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[1]+tf.expand_dims(self.M_qa_real,-1))-tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[1]+tf.expand_dims(self.M_qa_real,-1))
        # self.qa_real_2 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[2]+tf.expand_dims(self.M_qa_real,-1))-tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[2]+tf.expand_dims(self.M_qa_real,-1))
        # self.qa_real_3 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[3]+tf.expand_dims(self.M_qa_real,-1))-tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[3]+tf.expand_dims(self.M_qa_real,-1))
        # self.qa_imag_0 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[0]+tf.expand_dims(self.M_qa_imag,-1))+tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[0]+tf.expand_dims(self.M_qa_imag,-1))   
        # self.qa_imag_1 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[1]+tf.expand_dims(self.M_qa_imag,-1))+tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[1]+tf.expand_dims(self.M_qa_imag,-1))  
        # self.qa_imag_2 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[2]+tf.expand_dims(self.M_qa_imag,-1))+tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[2]+tf.expand_dims(self.M_qa_imag,-1)) 
        # self.qa_imag_3 = tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[3]+tf.expand_dims(self.M_qa_imag,-1))+tf.nn.relu(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[3]+tf.expand_dims(self.M_qa_imag,-1))   
        self.qa_real_0 = (self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[0])-(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[0])
        self.qa_real_1 = (self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[1])-(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[1])
        self.qa_real_2 = (self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[2])-(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[2])
        self.qa_real_3 = (self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_real)[3])-(self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_imag)[3])
        self.qa_imag_0 = (self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[0])+(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[0])
        self.qa_imag_1 = (self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[1])+(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[1])
        self.qa_imag_2 = (self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[2])+(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[2])
        self.qa_imag_3 = (self.narrow_convolution(tf.expand_dims(self.M_qa_imag,-1),self.kernels_real)[3])+(self.narrow_convolution(tf.expand_dims(self.M_qa_real,-1),self.kernels_imag)[3] )
    def max_pooling(self,conv):
        pooled = tf.nn.max_pool(
                    conv,
                    # ksize = [1, 8, 8, 1],
                    ksize = [1, 1, 1, 1],
                    # ksize = [1, 6, 6, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name = "pool")
        return pooled
    def avg_pooling(self,conv):
        pooled = tf.nn.avg_pool(
                    conv,
                    ksize = [1, self.q_length, self.a_length, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name = "pool")
        return pooled
    def pooling_graph(self):
        with tf.name_scope('pooling'):

      
            # raw_pooling_real = tf.contrib.layers.flatten(tf.reduce_mean(self.qa_real,1))
            # col_pooling_real = tf.contrib.layers.flatten(tf.reduce_mean(self.qa_real,2))
            # self.represent_real = tf.concat([raw_pooling_real,col_pooling_real],1)
            

            self.represent_real_0 = self.max_pooling(self.qa_real_0)
            self.represent_real_1 = self.max_pooling(self.qa_real_1)
            self.represent_real_2 = self.max_pooling(self.qa_real_2)
            self.represent_real_3 = self.max_pooling(self.qa_real_3)
            # print(self.represent_real)
            self.represent_real_0 =tf.reshape(self.represent_real_0 ,[self.batch_size,-1])
            self.represent_real_1 =tf.reshape(self.represent_real_1 ,[self.batch_size,-1])
            self.represent_real_2 =tf.reshape(self.represent_real_2 ,[self.batch_size,-1])
            self.represent_real_3 =tf.reshape(self.represent_real_3 ,[self.batch_size,-1])
            # print(self.represent_real)
            self.represent_img_0 = self.max_pooling(self.qa_imag_0)
            self.represent_img_1 = self.max_pooling(self.qa_imag_1)
            self.represent_img_2 = self.max_pooling(self.qa_imag_2)
            self.represent_img_3 = self.max_pooling(self.qa_imag_3)
            # self.represent_img=tf.reshape(self.represent_img,[self.batch_size,-1])
            # print(self.represent_img)
            self.represent_img_0 =tf.reshape(self.represent_img_0 ,[self.batch_size,-1])
            self.represent_img_1 =tf.reshape(self.represent_img_1 ,[self.batch_size,-1])
            self.represent_img_2 =tf.reshape(self.represent_img_2 ,[self.batch_size,-1])
            self.represent_img_3 =tf.reshape(self.represent_img_3 ,[self.batch_size,-1])
            w_0 = tf.Variable(tf.zeros(shape = [self.batch_size,tf.shape(self.represent_real_0)[-1]],name = 'W'))
            w_1 = tf.Variable(tf.zeros(shape = [self.batch_size,tf.shape(self.represent_real_1)[-1]],name = 'W'))
            w_2 = tf.Variable(tf.zeros(shape = [self.batch_size,tf.shape(self.represent_real_2)[-1]],name = 'W'))
            w_3 = tf.Variable(tf.zeros(shape = [self.batch_size,tf.shape(self.represent_real_3)[-1]],name = 'W'))
            # w = tf.Variable(tf.zeros(shape = [1,141120],name = 'W'))
            self.represent_real_W_0=tf.multiply(tf.multiply(self.represent_real_0,w_0),self.represent_img_0)
            self.represent_real_W_1=tf.multiply(tf.multiply(self.represent_real_1,w_1),self.represent_img_1)
            self.represent_real_W_2=tf.multiply(tf.multiply(self.represent_real_2,w_2),self.represent_img_2)
            self.represent_real_W_3=tf.multiply(tf.multiply(self.represent_real_3,w_3),self.represent_img_3)
            # print(self.represent_real_W)
            # raw_pooling_imag = tf.contrib.layers.flatten(tf.reduce_mean(self.qa_imag,1))
            # col_pooling_imag = tf.contrib.layers.flatten(tf.reduce_mean(self.qa_imag,2))
            # self.represent_imag = tf.concat([raw_pooling_imag,col_pooling_imag],1)
            self.represent = tf.concat([self.represent_real_0,self.represent_real_1,self.represent_real_2,self.represent_real_3,
                                        self.represent_img_0,self.represent_img_1,self.represent_img_2,self.represent_img_3,
                                        self.represent_real_W_0,self.represent_real_W_1,self.represent_real_W_2,self.represent_real_W_3],1)
            # self.represent = tf.concat([self.represent_real,self.represent_img,self.represent_real_W],1)
            #self.represent=tf.nn.dropout(self.represent, 0.4, name="hidden_output_drop")

            print(self.represent)

    def wide_convolution(self,embedding,kernel):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    kernel[i],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name="conv-1"
            )
            cnn_outputs.append(conv)
        cnn_reshaped = tf.concat(cnn_outputs,3)
        return cnn_reshaped
    def narrow_convolution(self,embedding,kernel):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    kernel[i],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
            )
            cnn_outputs.append(conv)
        # cnn_reshaped = tf.concat(cnn_outputs,3)
        # return cnn_outputs
        return cnn_outputs[0], cnn_outputs[1] , cnn_outputs[2] , cnn_outputs[3]
    def build_graph(self):
        self.create_placeholder()
        self.Embedding()
        self.convolution()
        self.pooling_graph()
        self.feed_neural_work()
        self.create_loss()