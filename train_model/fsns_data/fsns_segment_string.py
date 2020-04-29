import argparse
import os
import errno
import sys
import random
import json
import time


def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(
        description="生成用户字符串识别的切分字符串"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default="output/"
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="When set, this argument uses a specified text file as source for the text",
        default="",
        required=True
    )
    parser.add_argument(
        "-mi",
        "--min_char_count",
        type=int,
        nargs="?",
        help="The minimum number of characters per line, Default is 3.",
        default=3,

    )
    parser.add_argument(
        "-ma",
        "--max_char_count",
        type=int,
        nargs="?",
        help="The maximum number of characters per line, Default is 20.",
        default=20,
    )
    return parser.parse_args()


def generate_dic_txt(lines, output_dir):
    """
    :param lines:
    :param output_dir:
    :return:
    """
    dict_txt_file = os.path.join(output_dir, 'dic.txt')
    char_set = set(''.join(lines))

    index = 0
    new_lines = []
    for char in char_set:
        if char in [' ', '\t' , '\n', '\r']:
            continue
        new_lines.append('{}\t{}'.format(index, char))
        index += 1

    with open(dict_txt_file, 'w', encoding='utf-8') as f:
        for new_line in new_lines:
            f.write(new_line+'\n')

    print('【输出】生成 dic.txt 文件  输出路径{}, 文件行数 {}.'.format(dict_txt_file, index))


def combined_line(output_dir,
                  lines, min_chars_count, max_chars_count):
    """
    :param lines:
    :param output_dir
    :param min_chars_count:
    :param max_chars_count:
    :return:
    """
    random.seed(42)
    new_lines = []
    for line in lines:
        while True:
            line = line.replace(' ', '').replace('\r', '').replace('\n', '').replace('\t', '')
            count = random.randint(min_chars_count, max_chars_count)
            if len(line) > count:
                new_lines.append(line[0:count])
                line = line[count:]
            else:
                break

    output_file = os.path.join(output_dir, 'text_split.txt')

    with open(output_file, 'w',  encoding='utf-8') as f:
        for new_line in new_lines:

            f.write(new_line+'\n')
    print("Write {} lines in file [{}]".format(len(new_lines) , output_file))
    return output_file, len(new_lines)


def main():
    time_start = time.time()
    # Argument parsing
    args = parse_arguments()

    # Create the directory if it does not exist.
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    assert os.path.exists(args.input_file), \
        "Input file[{}] is not exists.".format(args.input_file)

    assert args.min_char_count <= args.max_char_count, \
        "min_char_count({}) must be less than max_char_count({})".format(args.min_char_count, args.max_char_count)

    print('Input file: {}'.format(args.input_file))

    print('Output dir: {} '.format(args.output_dir))
    print('MinCharsCount={} , MaxCharsCount={}'.format(args.min_char_count, args.max_char_count))

    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    generate_dic_txt(lines, args.output_dir)

    output_seg_file, count = combined_line(args.output_dir,  lines, args.min_char_count, args.max_char_count)
    print('【输出】{} 文件分割完成, 输出路径{}, 文件行数 {}.'.format(args.input_file, output_seg_file, count))
    time_elapsed = time.time() - time_start
    print('The code run {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    main()




