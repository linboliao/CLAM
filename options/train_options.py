from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
        self.parser.add_argument('--gpus', type=int, default=6, help="GPU indices ""comma separated, e.g. '0,1' ")
        self.parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
        self.parser.add_argument('--feat_model', type=str, default='uni_v1', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
        self.parser.add_argument('--clf_model', type=str, default='ABMIL', help='What model architecture to use.')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
        self.parser.add_argument('--lr_decay', type=bool, default=True, help='')
        self.parser.add_argument('--feat_size', type=int, default=1024)
        self.parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train (default: 200)')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Dataloaders batch size.')