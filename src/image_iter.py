from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import numbers
import math
import sklearn
import datetime
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_preprocess
import multiprocessing

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio

logger = logging.getLogger()
from utils import init_random
from augments.common import common_aug

class FaceImageIter(io.DataIter):
    def __init__(self, batch_size, data_shape,
                 path_imgrecs = None,
                 shuffle=False, aug_list=None, mean = None,
                 rand_mirror = False, cutout = None, crop = None, mask = None, gridmask = None,
                 downsample_back = 0.0, motion_blur = 0.0,
                 mx_model = None,
                 data_names=['data'], label_name='softmax_label', **kwargs):
        super(FaceImageIter, self).__init__()
        assert path_imgrecs
        self.kwargs = kwargs
        if path_imgrecs:
            self.rec_num = len(path_imgrecs)
            self.seq, self.oseq = [], []
            self.imgrec, self.header0, self.imgidx, self.id2range, self.seq_identity = [], [], [], [], []
            for path_imgrec in path_imgrecs:
                logging.info('loading recordio %s...',
                             path_imgrec)
                path_imgidx = path_imgrec[0:-4]+".idx"
                self.imgrec.append( recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r'))  # pylint: disable=redefined-variable-type
                s = self.imgrec[-1].read_idx(0)
                header, _ = recordio.unpack(s)
                if header.flag>0:
                  print('header0 label', header.label)
                  self.header0.append( (int(header.label[0]), int(header.label[1])))
                  #assert(header.flag==1)
                  self.imgidx.append(range(1, int(header.label[0])))
                  self.id2range.append( {} )
                  self.seq_identity.append(range(int(header.label[0]), int(header.label[1])))
                  for identity in self.seq_identity[-1]:
                    s = self.imgrec[-1].read_idx(identity)
                    header, _ = recordio.unpack(s)
                    a,b = int(header.label[0]), int(header.label[1])
                    self.id2range[-1][identity] = (a,b)
                    count = b-a
                  print('id2range', len(self.id2range[-1]))
                else:
                  self.imgidx.append(list(self.imgrec[-1].keys))

                self.seq.append(list(self.imgidx[-1]))
                self.oseq.append(self.imgidx[-1])
                print(len(self.seq[-1]))

        self.iteration = 0
        self.margin_policy = self.kwargs.get('margin_policy', 'step')
        self.use_bgr = self.kwargs.get('use_bgr', False)

        self.mx_model = mx_model

        self.check_data_shape(data_shape)
        if crop is not None:
            crop_h, crop_w = crop.crop_h, crop.crop_w
            data_shape = (data_shape[0], crop_h, crop_w)
        if 'loss_type' in self.kwargs and self.kwargs['loss_type'] == 6:
          self.provide_data = [(data_names[0], (batch_size,) + data_shape), (data_names[1], (batch_size,))]
        else:
          self.provide_data = [(data_names[0], (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.augs = common_aug(rand_mirror = rand_mirror, cutout = cutout, crop = crop,
                  mask = mask, gridmask = gridmask, downsample_back = downsample_back, motion_blur = motion_blur, mean = mean)

        print('rand_mirror: {}'.format( rand_mirror))
        self.provide_label = [(label_name, (batch_size, self.rec_num))]
        #self.provide_label = [(label_name, (batch_size, self.rec_num))] if self.rec_num > 1 else [(label_name, (batch_size, ))]
        #print(self.provide_label[0][1])
        self.cur = [0] * len(path_imgrecs)
        self.nbatch = 0
        self.is_init = False


    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = [0] * self.rec_num
        if self.shuffle:
          for i in range(self.rec_num):
            random.shuffle(self.seq[i])
        for i in range(self.rec_num):
          if self.seq[i] is None and self.imgrec[i] is not None:
              self.imgrec[i].reset()

    def num_samples(self, dataset_idx):
      return len(self.seq[dataset_idx])

    def margin(self, iteration):
        if self.margin_policy == 'fixed':
          m= self.kwargs['margin_m']
        elif self.margin_policy == 'step':
          steps = [100000, 300000, 1000000]
          values = [0, 0.35, 0.5]
          for i, step in enumerate(steps):
            if step > iteration:
              m = values[i]
              break
        elif self.margin_policy == 'linear':
            m = self.kwargs['margin_m'] / self.kwargs['max_steps'] * iteration
        return m

    def next_sample(self, dataset_idx):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq[dataset_idx] is not None:
          while True:
            if self.cur[dataset_idx] >= len(self.seq[dataset_idx]):
                raise StopIteration
            idx = self.seq[dataset_idx][self.cur[dataset_idx]]
            self.cur[dataset_idx] += 1
            if self.imgrec[dataset_idx] is not None:
              try:
                s = self.imgrec[dataset_idx].read_idx(idx)
              except:
                continue
              header, img = recordio.unpack(s)
              label = header.label
              if not isinstance(label, numbers.Number):
                label = label[0]
              return label, img, None, None
            else:
              label, fname, bbox, landmark = self.imglist[dataset_idx][idx]
              return label, self.read_image(fname), bbox, landmark
        else:
            s = self.imgrec[dataset_idx].read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            label = header.label
            if not isinstance(label, numbers.Number):
              label = label[0]
            return label, img, None, None

    def next(self):
        if not self.is_init:
          self.reset()
          self.is_init = True
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        self.nbatch+=1
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_margin = nd.empty((batch_size,))
        if self.provide_label is not None:
          batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            dataset_idx = 0
            while i < batch_size:
                batch_margin[i] = self.margin(self.iteration)
                _label, s, bbox, landmark = self.next_sample(dataset_idx)
                if(len(self.seq) > 1):
                  label = np.ones([self.rec_num,]) * (-1)
                  label[dataset_idx] = _label
                else:
                  label = _label
                dataset_idx = (dataset_idx + 1) % self.rec_num
                _data = self.imdecode(s)

                _data = self.augs.apply(_data)

                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue

                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration

        #db = mx.io.DataBatch(data=(batch_data,))
        #self.mx_model.forward(db, is_train=False)
        #net_out = self.mx_model.get_outputs()

        #print(net_out)
        self.iteration += 1
        return io.DataBatch([batch_data], [batch_label], batch_size - i)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s) #mx.ndarray
        return img

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        if self.use_bgr:
          return nd.transpose(datum[:, :, ::-1], axes=(2, 0, 1))
        else:
          return nd.transpose(datum, axes=(2, 0, 1))

class FaceImageIterList(io.DataIter):
  def __init__(self, iter_list):
    assert len(iter_list)>0
    self.provide_data = iter_list[0].provide_data
    self.provide_label = iter_list[0].provide_label
    self.iter_list = iter_list
    self.cur_iter = None

  def reset(self):
    self.cur_iter.reset()

  def next(self):
    self.cur_iter = random.choice(self.iter_list)
    while True:
      try:
        ret = self.cur_iter.next()
      except StopIteration:
        self.cur_iter.reset()
        continue
      return ret

if __name__ == '__main__':
  import config
  train_dataiter = FaceImageIter(
      batch_size           = 32,
      data_shape           = (3, 112, 112),
      path_imgrecs         = ['../datasets/faces_emore/train.rec'],
      shuffle              = True,
      rand_mirror          = True,
      mean                 = None,
      cutout               = config.cutout,
      crop                 = config.crop,
      mask                 = config.mask,
      gridmask             = config.gridmask,
      data_names           = ['data'],
      downsample_back      = config.config.downsample_back,
      motion_blur          = config.config.motion_blur,
      use_bgr              = config.config.use_bgr
  )

  batch = train_dataiter.next()
  data = batch.data[0].asnumpy()
  for i in range(32):
    img_data = data[i, ...].transpose([1, 2, 0])[:, :, ::-1]
    cv2.imwrite('temp/%d.png' % i, img_data)
