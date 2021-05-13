import tensorflow as tf
import numpy as np
import os
import time
import datetime
from helper_my import load_ms,load_wiki,get_overlap_dict, batch_gen_with_point_wise, load, prepare, batch_gen_with_single,test_my,batch_gen_with_point_MS
import operator
# from QA_CNN_point import QA_quantum as model
# from CNNQLM_II_flat import QA_quantum as model
# from vocab import QA_quantum as model
from dim   import QA_quantum as model
# from CNNQLM_I import QA_quantum as moedl
import random
import evaluation
import pickle
import config
from functools import reduce
from operator import mul
from tqdm import tqdm	

os.environ["CUDA_VISIBLE_DEVICES"]="0"



def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
now = int(time.time())
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print (timeStamp)

from functools import wraps

FLAGS = config.flags.FLAGS
# FLAGS._parse_flags()
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(("{}={}".format(attr.upper(), value)))
log_dir = 'wiki_log/' + timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/test_' + FLAGS.data + timeStamp
para_file = log_dir + '/test_' + FLAGS.data + timeStamp + '_para'
AP_dir = data_file + 'AP_'  + 'precise'
precision = data_file + 'precise'


def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco
#@log_time_delta
def predict(sess, cnn, test, alphabet, batch_size, q_len, a_len):
    scores = []
    # d = get_overlap_dict(test, alphabet, q_len, a_len)
    for data in batch_gen_with_single(test, alphabet, batch_size, q_len, a_len, overlap_dict=None):
        feed_dict = {
            cnn.question: data[0],
            cnn.answer: data[1],
            cnn.q_position:data[2],
            cnn.a_position:data[3],
            cnn.overlap:data[4],
            cnn.q_overlap:data[5],
            cnn.a_overlap:data[6]
        }
        score = sess.run(cnn.scores, feed_dict)
        scores.extend(score)
    return np.array(scores[:len(test)])

@log_time_delta
def test_point_wise():
    train, dev, test = load_ms(FLAGS.data, filter=FLAGS.clean) #ms
    # train, dev, test = load_wiki(FLAGS.data, filter=FLAGS.clean)  #wiki
    # train, test, dev = load(FLAGS.data, filter=FLAGS.clean) #trec
    q_max_sent_length = FLAGS.max_len_left
    a_max_sent_length = FLAGS.max_len_right
    print(q_max_sent_length)
    print(a_max_sent_length)
    print(len(train))
    print ('train question unique:{}'.format(len(train['question'].unique())))
    print ('train length', len(train))
    print ('test length', len(test))
    print ('dev length', len(dev))

    alphabet, embeddings,embeddings_complex = prepare(
        [train, test, dev], max_sent_length=a_max_sent_length,dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=True)
    # print(embeddings_complex)
    print ('alphabet:', len(alphabet))
    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            session_conf = tf.ConfigProto()
            session_conf.allow_soft_placement = FLAGS.allow_soft_placement
            session_conf.log_device_placement = FLAGS.log_device_placement
            session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default(), open(precision, "w") as log:
            s='num_filters: '+ str(FLAGS.num_filters) + '\n' + 'embedding_dim:  '+str(FLAGS.embedding_dim)+'\n'+'dropout_keep_prob:  '+str(FLAGS.dropout_keep_prob)+'\n'+'l2_reg_lambda:  '+str(FLAGS.l2_reg_lambda)+'\n'+'learning_rate:  '+str(FLAGS.learning_rate)+'\n'+'batch_size:  '+str(FLAGS.batch_size)+'\n''trainable:  '+str(FLAGS.trainable)+'\n'+'num_epochs:  '+str(FLAGS.num_epochs)+'\n''data:  '+str(FLAGS.data)+'\n'
            log.write(str(s) + '\n')
            Writer = open(AP_dir,'w') 
            # train,test,dev = load("trec",filter=True)
            # alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
            cnn = model(
                max_input_left=q_max_sent_length,
                max_input_right=a_max_sent_length,
                vocab_size=len(alphabet),
                embedding_size=FLAGS.embedding_dim,
                batch_size=FLAGS.batch_size,
                embeddings=embeddings,
                embeddings_complex=embeddings_complex,
                dropout_keep_prob=FLAGS.dropout_keep_prob,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                is_Embedding_Needed=True,
                trainable=FLAGS.trainable,
                overlap_needed=FLAGS.overlap_needed,
                position_needed=FLAGS.position_needed,
                pooling=FLAGS.pooling,
                hidden_num=FLAGS.hidden_num,
                extend_feature_dim=FLAGS.extend_feature_dim)
            cnn.build_graph()
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            starter_learning_rate = FLAGS.learning_rate
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate, global_step, 100, 0.96)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            # optimizer =  tf.train.GradientDescentOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            sess.run(tf.global_variables_initializer())
            map_max = 0.
            now = int(time.time())
            timeArray = time.localtime(now)
            timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
            timeDay = time.strftime("%Y%m%d", timeArray)
            print (timeStamp)
            num_params = get_num_params()
            log.write('num_params = ' +str(num_params) +'\n')
            n_batch = int(len(train)//FLAGS.batch_size)
            for i in range(FLAGS.num_epochs):
                # d = get_overlap_dict(
                #     train, alphabet, q_len=q_max_sent_length, a_len=a_max_sent_length)
                print('len_train',len(train))
                # datas = batch_gen_with_point_wise(train, alphabet, FLAGS.batch_size,
                #                                   q_len=q_max_sent_length, a_len=a_max_sent_length)
                for i in range(n_batch):
                    data = batch_gen_with_point_MS(train, alphabet, FLAGS.batch_size,
                                                  q_len=q_max_sent_length, a_len=a_max_sent_length,now = i)
           #     num_params = get_num_params()
           #     log.write('num_params = '+str(num_params)+ '\n')
                    data = list(data)
                    # print(data[0][0])
                    # print(data[0][1])
                    feed_dict = {
                        cnn.question: data[0][0],
                        cnn.answer: data[0][1],
                        cnn.input_y: data[0][2],
                        cnn.q_position:data[0][3],
                        cnn.a_position:data[0][4],
                        cnn.overlap:data[0][5],
                        cnn.q_overlap:data[0][6],
                        cnn.a_overlap:data[0][7]
                    }

                    _, step, loss, accuracy, pred, scores,input_y= sess.run(
                        [train_op, global_step, cnn.loss, cnn.accuracy,cnn.predictions, cnn.scores,cnn.input_y],feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}  ".format( time_str, step, loss, accuracy))
                now = int(time.time())
                timeArray = time.localtime(now)
                timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
                timeDay = time.strftime("%Y%m%d", timeArray)
                print (timeStamp)
                predicted = predict(
                    sess, cnn, train, alphabet, FLAGS.batch_size, q_max_sent_length, a_max_sent_length)
                predicted_AP=np.argmax(predicted,1)
                AP_train = evaluation.evaluationBypandas_AP(train, predicted_AP)
                AP_train = AP_train.values.tolist()
                # print(predicted)
                map_train = evaluation.evaluationBypandas_MAP(train, predicted[:,-1])
                # print(map_train)
                mrr_train = evaluation.evaluationBypandas_MRR10(train, predicted[:,-1])
                mrr_train = mrr_train.values.tolist()
                mrr_train = np.mean(mrr_train[:10])

                predicted_test = predict(
                    sess, cnn, test, alphabet, FLAGS.batch_size, q_max_sent_length, a_max_sent_length)

                predicted_test_AP=np.argmax(predicted_test,1)
                AP_test = evaluation.evaluationBypandas_AP(test, predicted_test_AP)
                AP_test = AP_test.values.tolist()
                map_test = evaluation.evaluationBypandas_MAP(test, predicted_test[:,-1])
                mrr_test = evaluation.evaluationBypandas_MRR10(test, predicted_test[:,-1])
                mrr_test = mrr_test.values.tolist()
                mrr_test = np.mean(mrr_test[:10])

                if map_test > map_max:
                    map_max = map_test
                    timeStamp = time.strftime(
                        "%Y%m%d%H%M%S", time.localtime(int(time.time())))
                    folder = 'runs/' + timeDay
                    out_dir = folder + '/' + timeStamp + \
                        '__' + FLAGS.data + str(map_test)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    # save_path = saver.save(sess, out_dir)
                print ("{}:train epoch:map {} mrr@10 {}".format(i, map_train,mrr_train))
                line1 = " {}:epoch: map_train {} mrr@10_train {}".format(i, map_train,mrr_train)
                print ("{}:test epoch:map {} mrr@10 {}".format(i, map_test,mrr_test))
                line2 = " {}:epoch: map_test {}  mrr@10_test {}".format(i, map_test,mrr_test)
                log.write(line2+'\n')
                Writer.write(str(AP_test[:100]) + '\n')
                log.flush()
            log.close()
            Writer.close()


if __name__ == '__main__':
    # test_quora()
    if FLAGS.loss == 'point_wise':
        test_point_wise()
    # test_pair_wise()
    # test_point_wise()
