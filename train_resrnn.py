from adda.util import config_logging, collect_vars, batch_generator
from adda.models.resrecnet import resrecnet
from sklearn.metrics import f1_score
from adda.models.model import get_model_fn
import logging
import os
import random
from collections import deque
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def RESRNNTrainer(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                model, output, gpu = '0', iterations = 50, batch_size = 20, 
                display = 1, lr = 1e-4, snapshot =10, weights = None, 
                weights_end = None, weights_scope =None, solver='sgd', 
                seed=1234, n_classes = 4, stepsize = 10, 
                generate_classification_metrics=True):
                
    config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    
    if seed is None:
        seed = random.randrange(2 ** 32 - 2)
    logging.info('Using random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)
    
    # create path directory
    if not os.path.exists(output):
        os.mkdir(output)
        
    # obtain model 
    model_fn = get_model_fn(model)
    
    # reset model
    tf.reset_default_graph()
    
    # create variables
    X = tf.placeholder(tf.float32, shape=(None, X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]),name='Input')
    y = tf.placeholder(tf.int64)
    is_training = tf.placeholder(tf.bool, []) # flag to use batch or all data
    prob = tf.placeholder_with_default(1.0, shape=()) # dropout
    
    # create model: im_batch is shape of images
    layers = model_fn(X, n_classes=n_classes, is_training=is_training, prob=prob)
    classifier_net = layers["output"]
    
    # obtain loss function, first applies softmax to use cross entropy loss function
    classifier_loss = tf.losses.sparse_softmax_cross_entropy(y, classifier_net)
    
    # create learning rate variable
    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    
    if solver == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_var, 0.99)
    else:
        optimizer = tf.train.AdamOptimizer(lr_var)
    
    
    # step to update
    step_classifier = optimizer.minimize(classifier_loss)
    
    # task accuracy
    predictions = tf.argmax(classifier_net, -1)
    accuracies = tf.reduce_mean(tf.cast(tf.equal(predictions, y), tf.float32))
    
    
    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    
    
    if weights:
        var_dict = collect_vars(model, end=weights_end, prepend_scope=weights_scope)
        logging.info('Restoring weights from {}:'.format(weights))
        for src, tgt in var_dict.items():
            logging.info('    {:30} -> {:30}'.format(src, tgt.name))
        restorer = tf.train.Saver(var_list=var_dict)
        restorer.restore(sess, weights)
    
    # obtain vars that will be trained
    model_vars = collect_vars(model)
    
    # save variables
    saver = tf.train.Saver(var_list=model_vars)
    output_dir = os.path.join(output,'snapshot')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # list of losses 
    losses = deque(maxlen=10)
    list_accuracies = deque(maxlen=2)
    list_f1scores = deque(maxlen=2)
    
    # bar to indicate progress to iterations 
    bar = tqdm(range(iterations))
    bar.set_description('{} (lr: {:.0e})'.format(output, lr))
    bar.refresh()
    
    ntrain = X_train.shape[0]
    num_batch = ntrain//(batch_size)
    sess.run(lr_var.assign(lr))
    
    # Batch generators
    gen_source_batch = batch_generator([X_train, y_train], batch_size)
    
    # iterate on variable var
    for i in bar:
        
        for _ in range(num_batch):
            
            X0, y0 = gen_source_batch.__next__()
            
            # Data input
            feed = {X: X0, 
                    y: y0,
                    is_training:True,
                    prob:0.5}

            # Update feature extractor and Predictor
            _, predloss = \
                sess.run([step_classifier,
                        classifier_loss], 
                        feed_dict=feed)
            
        # add loss value
        losses.append(predloss)
            
        
        # display advance according to 
        if i % display == 0:
            # RUN ON VALID DATA
            cross_ent_val, valid_accuracy = sess.run([classifier_loss, accuracies], 
                    feed_dict={ X: X_valid, 
                                y: y_valid, 
                                is_training:False})
            # RUN ON TEST DATA
            
            test_accuracy, y_preds, = sess.run([accuracies, predictions], feed_dict={X: X_test, y:y_test, prob:1.0, is_training:False})
            test_f1scores = f1_score(y_test, y_preds, average="weighted")
            
            # add to list
            list_accuracies.append(test_accuracy)
            list_f1scores.append(test_f1scores)
            
            logging.info('{:20} {:10.4f}  (avg: {:10.4f})  acc_val:{:10.4f}  acc_test:{:10.4f}'.format('Iteration {}:'.format(i+1), 
                    predloss, np.mean(losses), valid_accuracy, test_accuracy))
            
        if stepsize is not None and (i + 1) % stepsize == 0:
            lr = sess.run(lr_var.assign(lr * 0.1))
            logging.info('Changed learning rate to {:.0e}'.format(lr))
            bar.set_description('{} (lr: {:.0e})'.format(output, lr))
            
        if (i + 1) % snapshot == 0:
            snapshot_path = saver.save(sess, output_dir, global_step=i + 1)
            logging.info('Saved snapshot to {}'.format(snapshot_path))
    
    list_metrics_classification = []
    
    if generate_classification_metrics:
        # CALCULATE CLASSIFICATION MEASURES ON TEST DATA
        test_accuracy, y_preds = sess.run([accuracies, predictions], feed_dict={X: X_test, y:y_test, prob:1.0, is_training:False})
        test_f1scores = f1_score(y_test, y_preds, average="weighted")
        
        # metrics to classification
        list_metrics_classification.append(test_accuracy)
        list_metrics_classification.append(test_f1scores)
        
        print("ACC:",test_accuracy)
        print("F1:",test_f1scores)
        print
        # SAVE predictions
        np.savetxt(output+"/predicted.csv", y_preds, delimiter=",")
    
    # SAVE predictions
    y_preds = sess.run([predictions], feed_dict={X: X_test, y: y_test, is_training:False})
    np.savetxt(output+"/predicted.csv", y_preds[0], delimiter=",")

    coord.request_stop()
    coord.join(threads)
    sess.close()
    
    return list_metrics_classification