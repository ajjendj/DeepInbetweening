#!usr/bin/env python

import time
import os
import glob
import scipy.misc
import numpy
import cv2

import tensorflow as tf
from viz_util import generate_output_imagegrid

import pympler

from mem_top import mem_top

import gc
import pprint

class TrainerOptions(object):
    def __init__(self,
                 batch_size=128,
                 save_every_n=1,
                 checkpoint_every_n=200,
                 summarize_every_n=200,
                 summarize_img_every_n=10,
                 checkpoint_subdir='_check',
                 summary_subdir='_sum',
                 image_subdir='_image',
                 auto_restore=False):
       self.batch_size = batch_size
       self.save_every_n = save_every_n
       self.checkpoint_every_n = checkpoint_every_n
       self.summarize_every_n = summarize_every_n
       self.summarize_img_every_n = summarize_img_every_n
       self.checkpoint_subdir = checkpoint_subdir
       self.summary_subdir = summary_subdir
       self.image_subdir = image_subdir
       self.auto_restore = auto_restore

# session
# optimizer
# loader
# graph
class Trainer(object):
    def __init__(self,
                 run_dir,
                 graph,
                 loss,
                 optimizer,
                 loader_train,
                 loader_test,
                 global_step,
                 opts=TrainerOptions()):
        self.run_dir = run_dir
        self.graph = graph
        self.loss = loss
        self.optimizer = optimizer
        #self.apply_grads = apply_grads
        #self.grads_and_vars = grads_and_vars
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.opts = opts
        self.step = global_step

        # for grad, var in self.grads_and_vars:
        #     tf.histogram_summary(var.name + '/gradient', grad)
        #     mean = tf.reduce_mean(grad)
        #     stddev = tf.sqrt(tf.reduce_sum(tf.square(grad - mean)))
        #     summary_name = '/' + var.name + '/gradient'
        #     tf.scalar_summary('mean' + summary_name, mean)
        #     tf.scalar_summary('stddev' + summary_name, stddev)
        #     tf.scalar_summary('max' + summary_name, tf.reduce_max(var))
        #     tf.scalar_summary('min' + summary_name, tf.reduce_min(var))

        self.batch_loss_summary = tf.scalar_summary("Loss/batch", self.loss)
        self.train_loss_summary = tf.scalar_summary("Loss/train", self.loss)
        self.test_loss_summary = tf.scalar_summary("Loss/test", self.loss)

        # Set up session
        self.sess = tf.Session()
        self.summaries = tf.merge_all_summaries()

        # Initialize variables
        init = tf.initialize_all_variables()
        self.sess.run(init)

        # Set up checkpointing and summary writing for tensorboard
        self.check_dir = os.path.join(run_dir, opts.checkpoint_subdir)
        self._ensure_dir_exists(self.check_dir)

        self.sum_dir = os.path.join(run_dir, opts.summary_subdir)
        self._ensure_dir_exists(self.sum_dir)

        self.image_dir = os.path.join(run_dir, opts.image_subdir)
        self._ensure_dir_exists(self.image_dir)

        self.saver = tf.train.Saver()
        self.summary_writer = tf.train.SummaryWriter(self.sum_dir, self.sess.graph)

        self.tr = pympler.tracker.SummaryTracker()
        self.logfile = open('logfile.txt', 'w')

        if opts.auto_restore:
            self.restore()

    def save(self):
        save_path = os.path.join(self.check_dir, 'my-model')
        self.saver.save(self.sess, save_path, global_step=self.step)


    def restore(self):
        ckpt = tf.train.get_checkpoint_state(self.check_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.check_dir, ckpt_name))
            return True
        else:
            print('Failed to load checkpoint from %s' % self.check_dir)

    def save_image(self, image, step):
        save_path = os.path.join(self.image_dir, 'image_%.8d.png' % step)
        scipy.misc.imsave(save_path, image)

    def output_memstats(self):
        all_objects = pympler.muppy.get_objects()
        sum1 = pympler.summary.summarize(all_objects)
        sum1 = numpy.asarray(sum1)
        num_objects = len(all_objects)
        size_objects = numpy.sum(sum1[:,2].astype(int))
        print >> self.logfile, ('Num of objects: %d | Size of all objects in KB: %d' % (num_objects, size_objects/1024.0))
        return

    def train_iteration(self):

        step = self.step.eval(session = self.sess)
        print "STEP: ", step

        if (step%100) == 0:
            print >> self.logfile, "STEP:", step,
            self.output_memstats()
            print >> self.logfile, "\n"

        self.start_time = time.time()

        a, b = self.loader_train.next_batch_tensors(self.opts.batch_size)

        feed_opt = { self.graph.input_tensor : a,
                 self.graph.truth : numpy.reshape(b, (-1, b.shape[1], b.shape[2], b.shape[3])),
                 self.graph.keep_prob: 1.0}

        self.sess.run(self.optimizer, feed_dict=feed_opt)

        if (step % 50) == 0:
            a_train, b_train = self.loader_train.train_sample_tensors(1000)
            a_test, b_test = self.loader_test.next_batch_tensors(32)

            #get training loss and test lost without dropout
            feed_train = { self.graph.input_tensor : a_train,
                 self.graph.truth : numpy.reshape(b_train, (-1, b.shape[1], b.shape[2], b.shape[3])),
                 self.graph.keep_prob: 1.0}

            feed_test = { self.graph.input_tensor : a_test,
                 self.graph.truth : numpy.reshape(b_test, (-1, b.shape[1], b.shape[2], b.shape[3])),
                 self.graph.keep_prob: 1.0}

            [batch_loss, batch_summ] = self.sess.run([self.loss, self.batch_loss_summary], feed_dict=feed_opt)
            self.summary_writer.add_summary(batch_summ, step)
            [train_loss, train_summ] = self.sess.run([self.loss, self.train_loss_summary], feed_dict=feed_train)
            self.summary_writer.add_summary(train_summ, step)
            [test_loss, test_summ] = self.sess.run([self.loss, self.test_loss_summary], feed_dict=feed_test)
            self.summary_writer.add_summary(test_summ, step)

            [outtest, alpha] = self.sess.run([self.graph.output, self.graph.alpha_mask], feed_dict=feed_test)
            [outtest_t, alpha_t] = self.sess.run([self.graph.output_t, self.graph.alpha_mask_t], feed_dict=feed_test)

            #Generate grid of output images and save to location
            output_grid_RGB = generate_output_imagegrid(a_test,
                                                        numpy.reshape(b_test, (-1, b.shape[1], b.shape[2], b.shape[3])),
                                                        outtest,
                                                        outtest_t,
                                                        mask=alpha[:,:,:,0],
                                                        mask_t=alpha_t[:,:,:,0])
            #Note: uncomment with correct destination
            #fname = '/Users/ajjoshi/Desktop/MemLeak/test-images' + str(step) + '.png'
            #cv2.imwrite(fname, output_grid_RGB)
            #scipy.misc.imsave(fname, output_grid_RGB)

            print ('Step %d: time %4.4f | batch loss %0.8f | Train L2-loss: %4.4f | Test L2-loss: %4.4f'
            % (step, time.time() - self.start_time, batch_loss, train_loss, test_loss))

        if (step % self.opts.summarize_every_n) == 0:

            [summary] = self.sess.run([self.summaries], feed_dict=feed_train)
            self.summary_writer.add_summary(summary, step)

        if (step % self.opts.checkpoint_every_n) == 0:

            self.save()

    def train_iteration_classification(self):
        step = self.step.eval(session = self.sess)

        self.start_time = time.time()

        a, b = self.loader_train.next_batch_tensors(self.opts.batch_size)
        feed = { self.graph.input_tensor : a,
                 self.graph.truth : b }
        # run one step of the optimizer
        [loss_before] = self.sess.run([self.loss], feed_dict=feed)
        _, loss, pred = self.sess.run([self.apply_grads, self.loss, self.graph.prediction], feed_dict=feed)

        correct_prediction = tf.equal(tf.argmax(tf.reshape(self.graph.truth,(-1,2)),1), tf.argmax(self.graph.prediction,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_acc = accuracy.eval(feed_dict=feed, session = self.sess)

        a_test, b_test = self.loader_test.next_batch_tensors(100)
        feed_test = { self.graph.input_tensor : a_test,
                 self.graph.truth : b_test }
        test_acc = accuracy.eval(feed_dict=feed_test, session = self.sess)

        if (step % 1) == 0:
            #print ('Mean of Ground Truth: ', numpy.mean(gt_to_print))
            print ('Step %d: time %4.4f, batch loss %0.8f -> %0.8f'
            % (step, time.time() - self.start_time, loss_before, loss))
            print ('Train accuracy: %4.2f | Test accuracy: %4.2f' % (train_acc, test_acc))

        if (step % self.opts.summarize_every_n) == 0:
            [summary] = self.sess.run([self.summaries], feed_dict=feed)
            self.summary_writer.add_summary(summary, step)

        if (step % self.opts.checkpoint_every_n) == 0:
            self.save()

        #if (step % self.opts.summarize_img_every_n) == 0:
        #    image = self.sess.run(tf.squeeze(self.graph.image))
        #    self.save_image(image, step)
        self.step += 1

    def _ensure_dir_exists(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
