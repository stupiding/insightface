import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data-dir', default='', help='training set directory')
  parser.add_argument('--prefix', default='../model/model', help='directory to save model.')
  parser.add_argument('--pretrained', default='', help='pretrained model to load')
  parser.add_argument('--ckpt', type=int, default=1, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--loss-type', type=str, default="arcface", help='loss type')
  parser.add_argument('--verbose', type=int, default=2000, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
  parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
  parser.add_argument('--network', default='fresnet,100', help='specify network')
  parser.add_argument('--width-mult', type=float, default=1, help="width-mult")
  parser.add_argument("--shake-drop", default=False, action="store_true" , help="whether use ShakeDrop")
  parser.add_argument("--avg-down", default=False, action="store_true" , help="whether use avg downsample")
  parser.add_argument('--version-se', type=int, default=0, help='whether to use se in network')
  parser.add_argument('--version-ibn', type=int, default=0, help='whether to use IBN in resnet')
  parser.add_argument('--version-input', type=int, default=1, help='network input config')
  parser.add_argument('--version-output', type=str, default='E', help='network embedding output config')
  parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config')
  parser.add_argument('--version-act', type=str, default='prelu', help='network activation config')
  parser.add_argument('--version-bn', default='bn', help='version of bn: bn, row, col')
  parser.add_argument('--pyramid-alpha', type=int, default=0, help='0 for resnet, otherwise for pyramid alpha')
  parser.add_argument('--use-deformable', type=int, default=0, help='use deformable cnn in network')
  parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
  parser.add_argument('--fc7-lr-mult', type=float, default=1.0, help='lr mult for fc7')
  parser.add_argument("--fc7-no-bias", default=False, action="store_true" , help="fc7 no bias flag")
  parser.add_argument('--use-global-stats', default=False, action='store_true', help='use global stats in bn')
  parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
  parser.add_argument('--per-batch-size', type=int, default=128, help='batch size in each context')
  parser.add_argument('--margin-m', type=float, default=0.5, help='margin for loss')
  parser.add_argument('--margin-policy', type=str, default='fixed', help='margin_m policy [fixed, step, linear]')
  parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
  parser.add_argument('--margin-a', type=float, default=1.0, help='')
  parser.add_argument('--margin-b', type=float, default=0.0, help='')
  parser.add_argument('--easy-margin', type=int, default=0, help='')
  parser.add_argument('--margin', type=int, default=4, help='margin for sphere')
  parser.add_argument('--beta', type=float, default=1000., help='param for sphere')
  parser.add_argument('--beta-min', type=float, default=5., help='param for sphere')
  parser.add_argument('--beta-freeze', type=int, default=0, help='param for sphere')
  parser.add_argument('--gamma', type=float, default=0.12, help='param for sphere')
  parser.add_argument('--power', type=float, default=1.0, help='param for sphere')
  parser.add_argument('--scale', type=float, default=0.9993, help='param for sphere')
  parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
  parser.add_argument('--cutout', type=int, default=0, help='cut off aug')
  parser.add_argument('--downsample-back', type=float, default=0.0, help='use downsample data augmentation')
  parser.add_argument('--motion-blur', type=float, default=0.0, help='motion blur aug')
  parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
  parser.add_argument("--parallel", default=False, action="store_true" , help="whether use model parallel")
  args = parser.parse_args()
  return args

