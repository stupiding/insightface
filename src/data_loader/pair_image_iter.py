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

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../common'))
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
                 mx_model = None, ctx_num = 1, 
                 data_names=['data'], label_name='softmax_label', **kwargs):
        super(FaceImageIter, self).__init__()
        assert len(path_imgrecs) == 2
        self.kwargs = kwargs

        if path_imgrecs:
            self.rec_num = len(path_imgrecs)
            self.imgrec, self.imgidx, self.id2range, self.seq_identity = {}, {}, {}, {}
            #self.imgrec, self.imgidx, self.id2range, self.seq_identity = [], [], [], []
            for path_imgrec in path_imgrecs:
                logging.info('loading recordio %s...',
                             path_imgrec)

                rec_name = os.path.basename(path_imgrec)[:-4]
                assert rec_name in ['pos', 'neg']

                path_imgidx = path_imgrec[0:-4]+".idx"
                self.imgrec[rec_name] = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

                s = self.imgrec[rec_name].read_idx(0)
                header, _ = recordio.unpack(s)
                if header.flag>0:
                  print('header0 label', header.label)
                  #assert(header.flag==1)
                  self.imgidx[rec_name] = range(1, int(header.label[0]))
                  self.id2range[rec_name] = {}
                  self.seq_identity[rec_name] = range(int(header.label[0]), int(header.label[1]))
                  for identity, idx in enumerate(self.seq_identity[rec_name]):
                    s = self.imgrec[rec_name].read_idx(idx)
                    header, _ = recordio.unpack(s)
                    start_idx, end_idx = int(header.label[0]), int(header.label[1])
                    self.id2range[rec_name][identity] = list(range(start_idx, end_idx))
                  print('id2range', len(self.id2range[rec_name]))
                else:
                  self.imgidx[rec_name] = list(self.imgrec[rec_name].keys())

            self.id_seq = list(self.id2range['pos'].keys())
            random.shuffle(self.id_seq)
            self.seq = []

        self.iteration = 0
        self.use_bgr = self.kwargs.get('use_bgr', False)

        self.mx_model = mx_model

        self.check_data_shape(data_shape)
        if crop is not None:
            crop_h, crop_w = crop.crop_h, crop.crop_w
            data_shape = (data_shape[0], crop_h, crop_w)

        self.batch_size = batch_size
        self.per_batch_size = self.batch_size // ctx_num
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.augs = common_aug(rand_mirror = rand_mirror, cutout = cutout, crop = crop,
                  mask = mask, gridmask = gridmask, downsample_back = downsample_back, motion_blur = motion_blur, mean = mean)

        self.provide_data = [(data_names[0], (batch_size,) + data_shape)]
        self.provide_label = [(label_name, (batch_size, 1))]

        self.cur = 0
        self.pair_cur = 0
        self.nbatch = 0
        self.need_init = True
        self.batch_person_num = 40

        self.POS, self.NEG = 1, 0

    def reset(self):
      print('reset')
      self.cur = 0
      self.pair_cur = 0
      random.shuffle(self.id_seq)
      self.select_pairs()

    def get_feats(self, ids):
      def _get_feats(ids_list, rec_name):
        feats = []
        batch_data = nd.zeros(self.provide_data[0][1])
        for start_idx in range(0, len(ids_list), self.batch_size):
          for i in range(self.batch_size):
            if start_idx + i >= len(ids_list):
              break
            _idx = ids_list[start_idx + i]
            s = self.imgrec[rec_name].read_idx(_idx)
            header, img = recordio.unpack(s)
            _data = self.imdecode(img)
            _data = self.augs.apply(_data)
            batch_data[i] = self.postprocess_data(_data)

          db = mx.io.DataBatch(data=(batch_data,))
          self.mx_model.forward(db, is_train=False)
          _feats = self.mx_model.get_outputs()[0]
          _feats = _feats / mx.nd.sqrt((_feats ** 2).sum(1).reshape([-1, 1]))
          feats.append(_feats.asnumpy())
        
        return np.concatenate(feats, 0)[:len(ids_list), :]

      pos_list, neg_list = [], []
      for pid in ids:
        pos_list += self.id2range['pos'][pid]
        neg_list += self.id2range['neg'][pid]

      pos_feats = _get_feats(pos_list, 'pos')
      neg_feats = _get_feats(neg_list, 'neg')
      return pos_feats, neg_feats, pos_list, neg_list

    def select_pairs(self):
      pair_end = self.pair_cur + self.batch_person_num
      if pair_end > len(self.id_seq):
        self.need_init = True
        raise StopIteration
      ids = self.id_seq[self.pair_cur:pair_end]
      self.pair_cur = pair_end
      pos_feats, neg_feats, pos_list, neg_list = self.get_feats(ids)

      pos_scores = np.dot(pos_feats, pos_feats.T)
      neg_scores = np.dot(pos_feats, neg_feats.T)

      hard_pos, hard_neg = [], []

      pre_pos_num, pre_neg_num = 0, 0
      for pid in ids:
        cur_pos_num = len(self.id2range['pos'][pid])
        cur_neg_num = len(self.id2range['neg'][pid])
        pos_end_idx = pre_pos_num + cur_pos_num
        neg_end_idx = pre_neg_num + cur_neg_num

        cur_scores = pos_scores[pre_pos_num:pos_end_idx, :]

        _pre_scores = cur_scores[:, :pre_pos_num]
        _post_scores = cur_scores[:, pos_end_idx:]
        #_pos_scores = cur_scores[:, pre_pos_num:pos_end_idx]
        _pos_scores = cur_scores[:1, pre_pos_num:pos_end_idx]
        _neg_scores = neg_scores[pre_pos_num:pos_end_idx, pre_neg_num:neg_end_idx]

        hard_idx = np.argsort(_pos_scores.flat)[:32]
        hard_loc = np.unravel_index(hard_idx, _pos_scores.shape)
        hard_loc = np.stack(hard_loc).T
        for i,j in hard_loc:
          i_idx = pos_list[pre_pos_num + i]
          j_idx = pos_list[pre_pos_num + j]
          
          if _pos_scores[i,j] > 0.6:
            continue
          hard_pos.append(('pos', i_idx, self.POS, _pos_scores[i,j]))
          hard_pos.append(('pos', j_idx, self.POS, _pos_scores[i,j]))

        negs = [(_pre_scores, 'pos', pos_list[0:pre_pos_num]),
                (_post_scores, 'pos', pos_list[pos_end_idx:]),
                (_neg_scores, 'neg', neg_list[pre_neg_num:neg_end_idx])
               ]
        for _scores, _neg_src, _neg_list in negs:
          hard_idx = np.argsort(_scores.flat)[-32:]
          hard_loc = np.unravel_index(hard_idx, _scores.shape)
          hard_loc = np.stack(hard_loc).T
          for i,j in hard_loc:
            i_idx = pos_list[pre_pos_num + i]
            j_idx = _neg_list[j]
            if _scores[i,j] < 0.4:
              continue
            hard_neg.append(('pos', i_idx, self.NEG, _scores[i,j]))
            hard_neg.append((_neg_src, j_idx, self.NEG, _scores[i,j]))
        pre_pos_num = pos_end_idx
        pre_neg_num = neg_end_idx

      hard_seq = hard_pos + hard_neg
      hard_idx = list(range(0, len(hard_seq), 2))
      random.shuffle(hard_idx)

      # Whole batch
      for start_idx in range(0, len(hard_idx), self.batch_size // 2):
        end_idx = start_idx + self.batch_size // 2
        if end_idx > len(hard_idx):
          break
        # batch on every gpu
        for per_start_idx in range(start_idx, end_idx, self.per_batch_size // 2):
          per_end_idx = per_start_idx + self.per_batch_size // 2
          for i in range(2):
            for idx in hard_idx[per_start_idx:per_end_idx]:
              self.seq.append(hard_seq[idx+i])
      print(self.pair_cur, len(self.seq), len(hard_pos), len(hard_neg), self.seq[0])
      
    def num_samples(self):
      return len(self.imgidx['pos'])

    def next_sample(self):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        while True:
          if self.cur >= len(self.seq):
            self.cur = 0
            self.seq = []
            self.select_pairs()
          
          rec_name, idx, pos_flag, score = self.seq[self.cur]
          #print(rec_name, idx, pos_flag, score)
          self.cur += 1
          s = self.imgrec[rec_name].read_idx(idx)
          header, img = recordio.unpack(s)
          label = header.label
          if not isinstance(label, numbers.Number):
            label = label[0]
          return label, img, pos_flag, score

    def next(self):
        if self.need_init:
          self.reset()
          self.need_init = False

        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        self.nbatch+=1
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        if self.provide_label is not None:
          batch_label = nd.empty(self.provide_label[0][1])

        i = 0
        try:
            while i < batch_size:
                _label, s, pos_flag, score = self.next_sample()
                label = pos_flag
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
        if self.motion_blur > 0 or self.downsample_back > 0:
            data = data.asnumpy()
            if self.motion_blur > 0:
                data = self.motion_aug(data)
            if self.downsample_back > 0:
                data = self.downsample_aug(data)
            return mx.nd.array(data)
        else:
            return data

        """
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        
        return data
        """

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
  from train_pairwise import get_module

  from utils.parser import parse_args
  args = parse_args()
  args.pretrained = '../models/gy50_ohem-r100/model,1'
  pretrain_model = get_module(args, [('data', (32, 3, 112, 112))])

  from easydict import EasyDict as edict
  crop = edict()
  crop.crop_h = 112
  crop.crop_w = 112
  crop.hrange = 0
  crop.wrange = 0

  train_dataiter = FaceImageIter(
      batch_size           = 64,
      data_shape           = (3, 112, 112),
      path_imgrecs         = ['../datasets/gy50_ohem/pos.rec', '../datasets/gy50_ohem/neg.rec'],
      shuffle              = True,
      rand_mirror          = True,
      mean                 = None,
      cutout               = None, #config.cutout,
      crop                 = crop,
      mask                 = None, #config.mask,
      gridmask             = None, #config.gridmask,
      data_names           = ['data'],
      downsample_back      = config.config.downsample_back,
      motion_blur          = config.config.motion_blur,
      use_bgr              = config.config.use_bgr,
      mx_model             = pretrain_model
  )

  batch = train_dataiter.next()
  data = batch.data[0].asnumpy()
  label = batch.label[0].asnumpy()
  print(label)
  for i in range(64):
    img_data = data[i, ...].transpose([1, 2, 0])[:, :, ::-1]
    cv2.imwrite('temp/%d.png' % i, img_data)
