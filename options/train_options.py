from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--n_threads_train', default=4, type=int, help='# threads for loading data')
        self._parser.add_argument('--num_iters_validate', default=1, type=int, help='# batches to use when validating')

        self._parser.add_argument('--num_epochs', type=int, default=10, help= 'train epochs')
        self._parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate set to train')
        self._parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
        self._parser.add_argument('--nepochs_no_decay', type=int, default=20, help='# of epochs at starting learning rate')
        self._parser.add_argument('--nepochs_decay', type=int, default=10, help='# of epochs to linearly decay learning rate to zero')
        self.is_train = True
