from utility import Utility
import argparse

util = Utility()


def parse_arguments():
    '''
    Method to parse the commandline arguments
    :return: argument object
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path',
                        default='/tmp',
                        type=str,
                        help='The path to the '
                             'caffe model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    model = util.getVggModel(args.model_path)
    if model is not None:
        print('Finally')