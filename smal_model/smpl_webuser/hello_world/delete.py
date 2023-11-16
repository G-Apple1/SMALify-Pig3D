import argparse
import os

def get_command_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to be clear', type = str, default=None)
    arg = parser.parse_args()
    print(arg)
    return arg

if __name__ == '__main__':
    command_args = get_command_args()
    file = os.listdir(command_args.path)
    for _file in file:
        os.remove(os.path.join(command_args.path, _file))
