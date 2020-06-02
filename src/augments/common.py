import cv2, random
import numpy as np
import mxnet as mx
from .gridmask import GridMask

class common_aug():
  def __init__(self, rand_mirror = False, cutout = None, crop = None,
                  mask = None, gridmask = None, downsample_back = 0.0, motion_blur = 0.0, mean = None):
    if motion_blur > 0:
      load_motion_kernel(self)

    self.nbatch = 0

    self.rand_mirror = rand_mirror
    self.cutout = cutout
    self.crop = crop
    self.mask = mask
    if gridmask is not None:
      self.gridmask = GridMask(d1=gridmask.d1, d2=gridmask.d2, rotate=gridmask.rotate,
                           ratio=gridmask.ratio, mode=gridmask.mode, prob=gridmask.prob)
    else:
      self.gridmask = None
    self.downsample_back = downsample_back
    self.motion_blur = motion_blur

    self.mean = mean
    self.nd_mean = None
    if self.mean:
      self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
      self.nd_mean = mx.nd.array(self.mean).reshape((1,1,3))

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

  def mirror_aug(self, _data):
    if self.rand_mirror:
      _rd = random.randint(0,1)
      if _rd==1:
        _data = mx.ndarray.flip(data=_data, axis=1)
    return _data

  def crop_aug(self, _data):
    if self.crop is not None:
      crop_h, crop_w = self.crop.crop_h, self.crop.crop_w
      hrange, wrange = self.crop.hrange, self.crop.wrange

      img_h, img_w = _data.shape[:2]
      assert crop_h <= img_h and crop_w <= img_w

      full_hrange, full_wrange = img_h - crop_h, img_w - crop_w
      if hrange == -1:
        h_off = random.randint(0, full_hrange)
      else:
        cur_range = min(full_hrange // 2, hrange)
        h_off = random.randint(-cur_range, cur_range) + full_hrange // 2
      if wrange == -1:
        w_off = random.randint(0, full_wrange)
      else:
        cur_range = min(full_wrange // 2, hrange)
        w_off = random.randint(-cur_range, cur_range) + full_wrange // 2
      _data = _data[h_off:h_off+crop_h, w_off:w_off+crop_w, :]
    return _data

  def cutout_aug(self, _data):
    if self.cutout is not None:
      cutout_ratio = self.cutout.ratio
      cutout_size = self.cutout.size
      cutout_mode = self.cutout.mode
      cutout_filler = self.cutout.filler
      if random.random() < cutout_ratio:
        if cutout_mode == 'fixed':
          None
        elif cutout_mode == 'uniform':
          cutout_size = random.randint(1, cutout_size)
        centerh = random.randint(0, _data.shape[0]-1)
        centerw = random.randint(0, _data.shape[1]-1)
        half = cutout_size//2
        starth = max(0, centerh-half)
        endh = min(_data.shape[0], centerh+half)
        startw = max(0, centerw-half)
        endw = min(_data.shape[1], centerw+half)
        _data = _data.astype('float32')
        if cutout_filler > 0:
          _data[starth:endh, startw:endw, :] = cutout_filler
        else:
          # random init
          _data[starth:endh, startw:endw, :] = random.random() * 255
    return _data

  def mask_aug(self, _data):
    if self.mask is not None:
      img_h, img_w = _data.shape[:2]

      mask_ratio = self.mask.ratio
      mask_size = int(img_h * self.mask.size)
      mask_value = self.mask.value
      if random.random() < mask_ratio:
        _data[-mask_size:, :, :] = mask_value
    return _data

  def apply(self, _data):
    self.nbatch+=1
    _data = _data.astype('float32')
    if self.rand_mirror:
      _data = self.mirror_aug(_data)

    if self.crop is not None:
      _data = self.crop_aug(_data)

    if self.cutout is not None:
      _data = self.cutout_aug(_data)

    if self.gridmask is not None:
      self.gridmask.set_prob(self.nbatch, 70000)
      _data = self.gridmask.process(_data)

    if self.mask is not None:
      _data = self.mask_aug(_data)

    if self.motion_blur > 0 or self.downsample_back > 0:
      data = _data.asnumpy()
      if self.motion_blur > 0:
        data = self.motion_aug(data)
      if self.downsample_back > 0:
        data = self.downsample_aug(data)
      _data = mx.nd.array(data)

    if self.nd_mean is not None:
      _data -= self.nd_mean
      _data *= 0.0078125
    return _data

if __name__ == '__main__':
  a = common_aug()
