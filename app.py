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

    parser.add_argument('--output_path',
                        default='/tmp',
                        type=str,
                        help='Path for storing the Keras model '
                             'weights')

    parser.add_argument('--proto_path',
                        default='/tmp',
                        type=str,
                        help='Path to the proto.txt file of the '
                             'caffe model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    model = util.getVggModel(args.model_path,
                             args.output_path,
                             args.proto_path)