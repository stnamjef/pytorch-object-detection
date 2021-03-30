from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # dataset path
    VOC_DIR = '../dataset/VOC2007'
    COCO_DIR = '../dataset/COCO'

    # data
    min_size = 800   # image resize
    max_size = 1200  # image resize
    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # training
    epoch = 6

    use_drop = False # use dropout in RoIHead

    # model
    load_path = None

    # Proposal layer
    NMS_THRESH = 0.7
    N_TRAIN_PRE_NMS = 12000
    N_TRAIN_POST_NMS = 2000
    N_TEST_PRE_NMS = 6000
    N_TEST_POST_NMS = 1000  # -> in FPN paper
    MIN_SIZE = 16

    # Proposal target layer
    N_SAMPLE_PROPOSAL = 512  # -> in FPN paper
    POS_RATIO_PROPOSAL = 0.25
    POS_IOU_THRESH_PROPOSAL = 0.5
    NEG_IOU_THRESH_HI_PROPOSAL = 0.5
    NEG_IOU_THRESH_LO_PROPOSAL = 0.0

    # Anchor target layer
    N_SAMPLE_ANCHOR = 256
    POS_IOU_THRESH_ANCHOR = 0.7
    NEG_IOU_THRESH_ANCHOR = 0.3
    POS_RATIO_ANCHOR = 0.5

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict(), sort_dicts=False)
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
