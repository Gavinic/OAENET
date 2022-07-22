from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self._parser.add_argument('--n_threads_train', default=1, type=int, help='# threads for loading data')
        self._parser.add_argument('--input_path', type=str, help='path to image')
        self._parser.add_argument('--output_dir', type=str, default='./output', help='output path')
        self.is_train = False
