import argparse
import os
import errno
import time

"""
格式转换
./3000/7/173_Ukase_81594.jpg 81594 -->  ./3000/7/173_Ukase_81594.jpg ukase
"""


def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(
        description="输入的文件"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        required=True
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="输入文件路径",
        required=True
    )
    return parser.parse_args()


def generate_char_map(input_file, output_dir):
    """
    :param lines:
    :param output_dir:
    :return:
    """

    new_lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\r', '').replace('\n', '')
            if len(line.split(' ')) != 2:
                print("ERROR  line = {} ".format(line))
                continue
            path = line.split(' ')[0]

            if path.startswith('./'):
                path = path[2:]
            new_line = '{} {}'.format(path, path.split('_')[1].lower())
            print(new_line)
            new_lines.append(new_line)

    output_file_path = os.path.join(output_dir,input_file.split('/')[-1])
    with open(output_file_path, 'w',  encoding='utf-8') as f:
        for new_line in new_lines:
            f.write(new_line+'\n')

    print('【输出】生成文件  输出路径{}, 文件行数 {}.'.format(output_file_path, len(new_lines)))


def main():
    time_start = time.time()
    # Argument parsing
    args = parse_arguments()
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Create the directory if it does not exist.
    generate_char_map(args.input_file, args.output_dir)

    time_elapsed = time.time() - time_start
    print('The code run {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    main()