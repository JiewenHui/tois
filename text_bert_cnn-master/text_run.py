import sys
import os
import time
from sklearn import metrics
from text_model import *
from loader import *
import config
from tqdm import tqdm
import evaluation
import metircs
FLAGS = config.flags.FLAGS
FLAGS.flag_values_dict()

os.environ["CUDA_VISIBLE_DEVICES"]="0"
METRICS_MAP = ['MAP',  'MRR',  'MRR@10']
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def evaluate(sess,dev_data):
    '''批量的形式计算验证集或测试集上数据的平均loss，平均accuracy'''
    data_len = 0
    total_loss = 0.0
    total_acc = 0.0
    for batch_ids, batch_mask, batch_segment, batch_label,question_length,answer_length in batch_iter(dev_data,config.batch_size):
        batch_len = len(batch_ids)
        data_len+=batch_len
        feed_dict = feed_data(batch_ids, batch_mask, batch_segment, batch_label,question_length,answer_length, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss/data_len, total_acc/data_len

def predict(sess,dev_data):
    scores = []
    for batch_ids, batch_mask, batch_segment, batch_label,question_length,answer_length in batch_iter(dev_data,config.batch_size,haha=False,df=dev_data):
        feed_dict = feed_data(batch_ids, batch_mask, batch_segment, batch_label,question_length,answer_length, 1.0)
        score = sess.run([model.scores], feed_dict=feed_dict)
        # print(score)
        scores.extend(score[0]) 
    # print(scores)
    return np.array(scores[:len(dev_data)]) 

# def evaluateMRR(sess,dev_data):
#     '''批量的形式计算验证集或测试集上数据的平均loss，平均accuracy'''
#     scores =[]
#     for batch_ids, batch_mask, batch_segment, batch_label,question_length,answer_length in batch_iter(dev_data,config.batch_size,haha=False,df=dev_data):

#         feed_dict = feed_data(batch_ids, batch_mask, batch_segment, batch_label,question_length,answer_length, 1.0)
#         score = sess.run([model.loss, model.scores,model.labels], feed_dict=feed_dict)
#         scores.extend(score)   

#     predicted =  np.array(scores[:len(dev_data)])   
#     mrr_10 = evaluation.evaluationBypandas_MRR10(dev_data, predicted[:,-1])  
#     return mrr_10
    


def feed_data(batch_ids, batch_mask, batch_segment, batch_label,question_length,answer_length,keep_prob):
    '''构建text_model需要传入的数据'''
    # print("aaaaaaaa")
    # print(batch_ids.shape)
    # print('bbbbbbbb')
    # print("aaaaaaaa")
    # print(batch_mask.shape)
    # print('bbbbbbbb')
    # print("aaaaaaaa")
    # print(batch_segment.shape)
    # print('bbbbbbbb')
    # print("aaaaaaaa")
    # print(batch_label.shape)
    # print('bbbbbbbb')
    # print("aaaaaaaa")
    # print(question_length.shape)
    # print('bbbbbbbb')
    # print("aaaaaaaa")
    # print(answer_length.shape)
    # print('bbbbbbbb')
 
    feed_dict = {
        model.input_ids: np.array(batch_ids),
        model.input_mask: np.array(batch_mask),
        model.segment_ids: np.array(batch_segment),
        model.labels: np.array(batch_label),
        model.q_length: np.array(question_length),
        model.a_length: np.array(answer_length),
        model.keep_prob:keep_prob
    }
    return feed_dict


def optimistic_restore(session, save_file):
    """载入bert模型"""
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for
                      var in tf.global_variables()
                      if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],tf.global_variables()),tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                # print("going to restore.var_name:",var_name,";saved_var_name:",saved_var_name)
                restore_vars.append(curr_var)
            else:
                print("variable not trained.var_name:",var_name)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def train():
    '''训练模型text_bert_cnn模型'''
    HAHA = True
    tensorboard_dir=os.path.join(config.output_dir, "tensorboard/textcnn")
    save_dir=os.path.join(config.output_dir, "checkpoints/textcnn")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    start_time = time.time()

    tf.logging.info("*****************Loading training data*****************")
    train_examples = TextProcessor().get_train_examples(config.data_dir)
    if not os.path.exists(config.data_dir + '/train.csv'):
    	print('hahahahahahah')
    	trian_data= convert_examples_to_features(train_examples, label_list, config.seq_length,tokenizer,name='train')
    tf.logging.info("*****************Loading dev data*****************")
    dev_examples = TextProcessor().get_dev_examples(config.data_dir)
    if not os.path.exists(config.data_dir + '/dev.csv'):
    	print('houhouhouhouhou')
    	dev_data = convert_examples_to_features(dev_examples, label_list, config.seq_length, tokenizer,name='dev')

    else:
    	print('heiheiheieheihei')
    	HAHA = False
    	trian_data,dev_data = load(config.data_dir)
    	print(trian_data.head())
    	# exit()
    tf.logging.info("Time cost: %.3f seconds...\n" % (time.time() - start_time))
    tf.logging.info("Building session and restore bert_model...\n")
    session = tf.Session()
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(session.graph)
    optimistic_restore(session, config.init_checkpoint)


    tf.logging.info('Training and evaluating...\n')
    best_acc= 0
    BEST_MRR = 0
    last_improved = 0  # record global_step at best_val_accuracy
    flag=False

    for epoch in range(config.num_epochs):
        batch_train = batch_iter(trian_data,config.batch_size,haha=HAHA,df=trian_data)
        start = time.time()
        tf.logging.info('Epoch:%d'%(epoch + 1))
        all_metrics = np.zeros(len(METRICS_MAP))
        for batch_ids, batch_mask, batch_segment, batch_label,question_length,answer_length in batch_train:
            feed_dict = feed_data(batch_ids, batch_mask, batch_segment, batch_label,question_length,answer_length, config.keep_prob)
            _, global_step, train_summaries, train_loss, train_accuracy = session.run([model.optim, model.global_step,
                                                                                    merged_summary, model.loss,
                                                                                    model.acc], feed_dict=feed_dict)


            if global_step % config.print_per_batch == 0:
                end = time.time()
                predicted = predict(session,dev_data)

                # print(predicted.shape)
                # print(dev_data.shape)

                # MRR_10 = evaluateMRR(session,dev_data)
                map_test = evaluation.evaluationBypandas_MAP(dev_data, predicted[:,-1])  
                MRR_10 =  evaluation.evaluationBypandas_MRR10(dev_data, predicted[:,-1])  
              
                if MRR_10 > BEST_MRR:
                    saver.save(session, save_path)
                    BEST_MRR = MRR_10
                    last_improved=global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                tf.logging.info("step: {},train loss: {:.3f}, train accuracy: {:.3f},training speed: {:.3f}sec/batch {}".format(
                        global_step, train_loss, train_accuracy, (end - start) / config.print_per_batch,improved_str))
                tf.logging.info("step: {},val map: {:.3f}, val mrr@10: {:.3f},training speed: {:.3f}sec/batch {}".format(
                        global_step,map_test, MRR_10,(end - start) / config.print_per_batch,improved_str))
                start = time.time()
                
            if global_step - last_improved > config.require_improvement:
                tf.logging.info("No optimization over 1500 steps, stop training")
                flag = True
                break
        if flag:
            print('BEST_MRR',BEST_MRR)
            tf.logging.info(BEST_MRR)
            break
        config.lr *= config.lr_decay
        print(BEST_MRR)


def test():
    '''testing'''

    save_dir = os.path.join(config.output_dir, "checkpoints/textcnn")
    save_path = os.path.join(save_dir, 'best_validation')

    if not os.path.exists(save_dir):
        tf.logging.info("maybe you don't train")
        exit()

    tf.logging.info("*****************Loading testing data*****************")
    test_examples = TextProcessor().get_test_examples(config.data_dir)
    test_data= convert_examples_to_features(test_examples, label_list, config.seq_length,tokenizer)

    input_ids,input_mask,segment_ids=[],[],[]

    for features in test_data:
        input_ids.append(features['input_ids'])
        input_mask.append(features['input_mask'])
        segment_ids.append(features['segment_ids'])

    config.is_training = False
    session=tf.Session()
    session.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    saver.restore(sess=session,save_path=save_path)

    tf.logging.info('Testing...')
    test_loss,test_accuracy = evaluate(session,test_data)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    tf.logging.info(msg.format(test_loss, test_accuracy))

    batch_size=config.batch_size
    data_len=len(test_data)
    num_batch=int((data_len-1)/batch_size)+1
    y_test_cls=[features['label_ids'] for features in test_data]
    y_pred_cls=np.zeros(shape=data_len,dtype=np.int32)


    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        feed_dict={
            model.input_ids: np.array(input_ids[start_id:end_id]),
            model.input_mask: np.array(input_mask[start_id:end_id]),
            model.segment_ids: np.array(segment_ids[start_id:end_id]),
            model.keep_prob:1.0,
        }
        y_pred_cls[start_id:end_id]=session.run(model.y_pred_cls,feed_dict=feed_dict)

    #evaluate
    tf.logging.info("Precision, Recall and F1-Score...")
    tf.logging.info(metrics.classification_report(y_test_cls, y_pred_cls, target_names=label_list))

    tf.logging.info("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    tf.logging.info(cm)


if __name__ == '__main__':

    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    tf.logging.set_verbosity(tf.logging.INFO)
    config = TextConfig()
    label_list = TextProcessor().get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file, do_lower_case=False)
    # model = TextCNN(config)
    alphabet = []
    with open(config.vocab_file) as f:
        for i,line in enumerate(f):
            pass
        count = i
    model = QA_quantum(
                config = config,
                max_input_left=20,
                max_input_right=100,
                vocab_size=count,
                embedding_size=FLAGS.embedding_dim,
                batch_size=FLAGS.batch_size,
                # embeddings=embeddings,
                # embeddings_complex=embeddings_complex,
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
    model.build_graph()
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
    else:
        exit()
