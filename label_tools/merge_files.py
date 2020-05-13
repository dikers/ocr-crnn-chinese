import argparse
import shutil
import errno
import time
import json
import glob
import os



class MergeFile(object):

    def __init__(self):
        args = self.parse_arguments()
        self.output_dir = args.output_dir
        self.input_dir = args.input_dir

    def parse_arguments(self):
        """
            Parse the command line arguments of the program.
        """

        parser = argparse.ArgumentParser(
            description="对图片中的文字区域进行裁剪"
        )
        parser.add_argument(
            "-o",
            "--output_dir",
            type=str,
            nargs="?",
            help="输出文件的本地路径",
            required=True
        )
        parser.add_argument(
            "-i",
            "--input_dir",
            type=str,
            nargs="?",
            help="输入文件路径",
            required=True
        )

        return parser.parse_args()


    def generate_char_map(self, lines, output_dir):

        """
        :param lines:
        :param output_dir:
        :return:
        """
        char_map_json_file = os.path.join(output_dir, 'char_map.json')
        temp_str = ''.join(lines).replace('\n', '')
        char_set = set(temp_str)

        single_char_map = {}
        index = 0
        for char in char_set:
            single_char_map[char] = index
            index += 1

        json_string = json.dumps(single_char_map, ensure_ascii=False, indent=1)
        with open(char_map_json_file, 'w', encoding='utf-8') as f:
            f.write(json_string)
        print('【输出】生成 Char map 文件  输出路径{}, 文件行数 {}.'.format(char_map_json_file, len(single_char_map)))


    def parse_file_list(self, input_dir):
        """
        """

        # print(" input dir: {} ".format(input_dir))
        dirs = os.listdir(input_dir)
        char_map_lines = []
        label_lines = []

        for index, temp_dir in enumerate(dirs):
            image_dir = os.path.join(input_dir, temp_dir)

            if os.path.isdir(image_dir):
                # print('index:  {}  [{}]'.format(index, image_dir))
                single_images_dir = os.path.join(image_dir, 'images')
                labels_file = os.path.join(image_dir, 'labels.txt' )
                print(labels_file)
                with open(labels_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        label_lines.append(line)
                        # print('** ', line.split('\t')[1].replace('\n', ''))
                        char_map_lines.append(line.split('\t')[1].replace('\n', ''))
                for image_file in os.listdir(single_images_dir):
                    print(os.path.join(single_images_dir, image_file))
                    shutil.copy(os.path.join(single_images_dir, image_file), os.path.join(self.output_dir, 'images'))

        self.generate_char_map(char_map_lines,  self.output_dir)

        total_label_file = os.path.join(self.output_dir, 'labels.txt')

        with open(total_label_file, 'w', encoding='utf-8') as tf:
            tf.writelines(label_lines)

        print('【输出】生成 Labels 文件  输出路径{}, 文件行数 {}.'.format(total_label_file, len(label_lines)))




    def main(self):
        time_start = time.time()
        # Argument parsing
        args = self.parse_arguments()
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)

        try:
            os.makedirs(args.output_dir)
            os.makedirs(os.path.join(args.output_dir, 'images'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if not os.path.exists(args.input_dir):
            print("输入路径不能为空  input_dir[{}] ".format(args.input_dir))
            return
        self.parse_file_list(args.input_dir)

        time_elapsed = time.time() - time_start
        print('The code run {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



if __name__ == "__main__":
    mergeFile = MergeFile()
    mergeFile.main()