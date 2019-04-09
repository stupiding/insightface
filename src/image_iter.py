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

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_preprocess
import multiprocessing

logger = logging.getLogger()


class FaceImageIter(io.DataIter):
    def __init__(self, batch_size, data_shape,
                 path_imgrecs = None,
                 shuffle=False, aug_list=None, mean = None,
                 rand_mirror = False, cutoff = 0,
                 downsample_back = 0.0, motion_blur = 0.0,
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

                if shuffle:
                  self.seq.append(self.imgidx[-1])
                  self.oseq.append(self.imgidx[-1])
                  print(len(self.seq[-1]))
                else:
                  self.seq.append(None)

        self.iteration = 0
        self.margin_policy = self.kwargs['margin_policy']
        self.mean = mean
        self.nd_mean = None
        if self.mean:
          self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
          self.nd_mean = mx.nd.array(self.mean).reshape((1,1,3))

        if motion_blur > 0:
          self.load_motion_kernel()

        self.check_data_shape(data_shape)
        if self.kwargs['loss_type'] == 6:
          self.provide_data = [(data_names[0], (batch_size,) + data_shape), (data_names[1], (batch_size,))]
        else:
          self.provide_data = [(data_names[0], (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.cutoff = cutoff
        self.downsample_back = downsample_back
        self.motion_blur = motion_blur
        #self.provide_label = [(label_name, (batch_size, self.rec_num))]
        self.provide_label = [(label_name, (batch_size, self.rec_num))] if self.rec_num > 1 else [(label_name, (batch_size, ))]
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

    def num_samples(self, data_idx):
      return len(self.seq[data_idx])

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

    def next_sample(self, data_idx):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq[data_idx] is not None:
          while True:
            if self.cur[data_idx] >= len(self.seq[data_idx]):
                raise StopIteration
            idx = self.seq[data_idx][self.cur[data_idx]]
            self.cur[data_idx] += 1
            if self.imgrec[data_idx] is not None:
              s = self.imgrec[data_idx].read_idx(idx)
              header, img = recordio.unpack(s)
              label = header.label
              if not isinstance(label, numbers.Number):
                label = label[0]
              return label, img, None, None
            else:
              label, fname, bbox, landmark = self.imglist[data_idx][idx]
              return label, self.read_image(fname), bbox, landmark
        else:
            s = self.imgrec[data_idx].read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img, None, None

    def brightness_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      src *= alpha
      return src

    def contrast_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      coef = np.array([[[0.299, 0.587, 0.114]]])
      gray = src * coef
      gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
      src *= alpha
      src += gray
      return src

    def saturation_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      coef = np.array([[[0.299, 0.587, 0.114]]])
      gray = src * coef
      gray = np.sum(gray, axis=2, keepdims=True)
      gray *= (1.0 - alpha)
      src *= alpha
      src += gray
      return src

    def color_aug(self, img, x):
      augs = [self.brightness_aug, self.contrast_aug, self.saturation_aug]
      random.shuffle(augs)
      for aug in augs:
        #print(img.shape)
        img = aug(img, x)
        #print(img.shape)
      return img

    def mirror_aug(self, img):
      _rd = random.randint(0,1)
      if _rd==1:
        for c in xrange(img.shape[2]):
          img[:,:,c] = np.fliplr(img[:,:,c])
      return img

    def load_motion_kernel(self):
      fs = cv2.FileStorage('resources/blur_kernels_13.xml', cv2.FILE_STORAGE_READ)
      kernel_number = int(fs.getNode('kernel_number').real())
      kernel_size = int(fs.getNode('kernel_size').real())
      kernel_prefix = fs.getNode('kernel_prefix').string()
      self.kernels = []
      for i in range(kernel_number):
        self.kernels.append(fs.getNode(kernel_prefix + '_' + str(i)).mat())
      
    def motion_aug(self, img):
      if random.random() < self.motion_blur:
        kernel_index = random.randint(0, len(self.kernels)-1)
        blurred_img = cv2.filter2D(img, -1, self.kernels[kernel_index])
        return blurred_img
      else:
        return img

    def downsample_aug(self, img):
      if random.random() < self.downsample_back:
        sizes = [(size, size) for size in range(32, 112, 16)][::-1]
        downsample_index = random.randint(0, len(sizes) - 1)
        downsampled_img = cv2.resize(img, sizes[downsample_index])
        return cv2.resize(downsampled_img, img.shape[:2])
      else:
        return img

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
            data_idx = 0
            while i < batch_size:
                batch_margin[i] = self.margin(self.iteration)
                _label, s, bbox, landmark = self.next_sample(data_idx)
                if(len(self.seq) > 1):
                  label = np.ones([self.rec_num,]) * (-1)
                  label[data_idx] = _label
                else:
                  label = _label
                data_idx = (data_idx + 1) % self.rec_num
                _data = self.imdecode(s)
                if self.rand_mirror:
                  _rd = random.randint(0,1)
                  if _rd==1:
                    _data = mx.ndarray.flip(data=_data, axis=1)
                _data = self.augmentation_transform(_data)
                if self.nd_mean is not None:
                    _data = _data.astype('float32')
                    _data -= self.nd_mean
                    _data *= 0.0078125
                if self.cutoff>0:
                  centerh = random.randint(0, _data.shape[0]-1)
                  centerw = random.randint(0, _data.shape[1]-1)
                  half = self.cutoff//2
                  starth = max(0, centerh-half)
                  endh = min(_data.shape[0], centerh+half)
                  startw = max(0, centerw-half)
                  endw = min(_data.shape[1], centerw+half)
                  _data = _data.astype('float32')
                  #print(starth, endh, startw, endw, _data.shape)
                  _data[starth:endh, startw:endw, :] = 127.5
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                #print('aa',data[0].shape)
                #data = self.augmentation_transform(data)
                #print('bb',data[0].shape)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration
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
        data = data.asnumpy()
        if self.motion_blur > 0:
            data = self.motion_aug(data)
        if self.downsample_back > 0:
            data = self.downsample_aug(data)
        return mx.nd.array(data)

        """
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        
        return data
        """

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
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


