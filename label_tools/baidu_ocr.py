from aip import AipOcr
import argparse
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
import argparse
import shutil
import errno
import time
import uuid
import fitz
import json
import glob
import cv2
import os


""" 你的 APPID AK SK """
APP_ID = '你的 App ID'
API_KEY = '你的 Api Key'
SECRET_KEY = '你的 Secret Key'


class PreProcessingImage(object):

    def __init__(self):
        args = self.parse_arguments()
        self.file_map = {}
        self.ocr_client = AipOcr(args.app_id,
                                 args.api_key,
                                 args.secret_key)

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
        parser.add_argument(
            "--app_id",
            type=str,
            nargs="?",
            help="你的 App ID",
            required=True
        )
        parser.add_argument(
            "--api_key",
            type=str,
            nargs="?",
            help="你的 Api Key",
            required=True
        )
        parser.add_argument(
            "--secret_key",
            type=str,
            nargs="?",
            help="你的 Secret Key",
            required=True
        )

        return parser.parse_args()


    def pyMuPDF_fitz(self, pdf_path, image_path, image_name):
        """
        pdf文件转换为png，存储到指定路径和指定名称
        :param pdf_path: 读入pdf文件路径
        :param image_path: 保存路径
        :param image_name: 保存名称
        """
        pdfDoc = fitz.open(pdf_path)
        for pg in range(pdfDoc.pageCount):
            page = pdfDoc[pg]
            rotate = int(0)
            # 每个尺寸的缩放系数为1.3，这将为我们生成分辨率提高2.6的图像。
            # 此处若是不做设置，默认图片大小为：792X612, dpi=96
            zoom_x = 3 #(1.33333333-->1056x816)   (2-->1584x1224)
            zoom_y = 3
            mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
            pix = page.getPixmap(matrix=mat, alpha=False)

            # pix = pdfDoc.getPagePixmap(pg)

            if not os.path.exists(image_path):#判断存放图片的文件夹是否存在
                os.makedirs(image_path) # 若图片文件夹不存在就创建

            pix.writePNG(image_path+'/'+'%s.png'%(image_name))#将图片写入指定的文件夹内

    def convert_pdf(self, pdf_path, save_path):
        """
        文件路径中pdf文件转换为png，转存到新的路径
        :param pdf_path: 读入文件路径
        :param savePath: 保存路径
        """
        if not os.path.exists(save_path):  # 保存路径不存在，则创建路径
            os.makedirs(save_path)

        if 'pdf' in pdf_path:
            save_name = pdf_path.split('/')[-1].replace(".pdf", "")
            self.pyMuPDF_fitz(pdf_path, save_path, 'image')
            print("Convert PDF to PNG finished!")
        else:
            print("Not a valid pdf file!")


    def _get_file_content(self, file_path):
        with open(file_path, 'rb') as fp:
            return fp.read()


    def call_ocr_image(self, image_path, output_path):
        """ 读取图片 """

        image = self._get_file_content(image_path)

        """ 如果有可选参数 """
        options = {}
        options["recognize_granularity"] = "big"
        options["probability"] = "true"
        options["accuracy"] = "normal"
        options["detect_direction"] = "true"

        """ 带参数调用通用票据识别 """
        results = self.ocr_client.receipt(image, options)

        print(results)

        json_str = json.dumps(results)
        with open(os.path.join(output_path, 'data.json'), 'w') as json_file:
            json_file.write(json_str)



    def read_local_image_file(self, input_dir, output_dir):
        """
        生成临时文件夹， 保存原始图片 json 和裁剪的图片
        :param input_dir:
        :param output_dir:
        :return:
        """
        # the tuple of file types
        #types = ('*.pdf', '*.jpg', '*.png', '*.jpeg')
        types = ('*.jpg', '*.png')
        files_grabbed = []
        for files in types:
            files_grabbed.extend(glob.glob(os.path.join(input_dir, files)))

        for file in files_grabbed:
            suid = ''.join(str(uuid.uuid4()).split('-'))

            file_output_dir = os.path.join(output_dir,
                                           file.split('/')[-1].replace('.', '_').replace('.', '') + '_' + suid[0:8])

            os.makedirs(file_output_dir)

            print("------------ file: ", file)
            print("------------ file_output_dir: ", file_output_dir)
            out_image_path = os.path.join(file_output_dir, 'image.png')
            if file.lower().endswith('pdf'):
                self.convert_pdf(file, file_output_dir)
            else:
                shutil.copy(file, out_image_path)

            self.file_map[file_output_dir] = out_image_path

        print(self.file_map)

    def parse_file_list(self):
        """
        调用Textract 进行文本识别
        :param s3_file_prefix:
        :return: json_file_list
        """
        if self.file_map is None or len(self.file_map) == 0:
            print("Warning:  没有要处理的图片")
            return

        for index, file_item in enumerate(self.file_map.items()):
            print('No: {}  Path: {}  file: {} '.format(index,  file_item[0], file_item[1]))
            self.call_ocr_image(file_item[1], file_item[0])
            self.create_gt_image_label(file_item[0])


    def create_gt_image_label(self, base_dir):
        json_file = os.path.join(base_dir, 'data.json')
        image_file = os.path.join(base_dir, 'image.png')


        with open(json_file, 'r') as f:
            data = json.load(f)

        print(len(data['words_result']))

        for item in data['words_result']:
            print(item['location'], item['words'])

        save_img = cv2.imread(image_file)
        save_path = os.path.join(base_dir, 'images')
        gt_labels_file = os.path.join(base_dir, 'labels.txt')
        print('---------------- json_file ', json_file)
        with open(json_file, 'r') as f:
            data = json.load(f)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        gt_labels = ""
        # save image bounding box/polygon the detected lines/text
        for i, block in enumerate(data['words_result']):
            # Draw box around entire LINE
            left = block['location']['left']
            top = block['location']['top']
            width = block['location']['width']
            height = block['location']['height']
            c_img = save_img[top: int(top + height), left: int(left + width)]
            name = base_dir.split('/')[-1]+'_'+str(i).zfill(3)+".png"

            f = os.path.join(save_path, name)
            gt_labels += ("{}\t{}\n".format(name, block['words']))
            Path(save_path).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f, c_img)
            print('Image Shape: {}'.format(c_img.shape))

        with open(gt_labels_file, "w") as f:
            f.write(gt_labels)
            print(' [{}] 文件保存成功 '.format(gt_labels_file))


    def main(self):
        time_start = time.time()
        # Argument parsing
        args = self.parse_arguments()
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)

        try:
            os.makedirs(args.output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if not os.path.exists(args.input_dir):
            print("输入路径不能为空  input_dir[{}] ".format(args.input_dir))
            return
        # step 1 . 生成文件夹
        self.read_local_image_file(args.input_dir, args.output_dir)



        # step 2. OCR
        self.parse_file_list()

        time_elapsed = time.time() - time_start
        print('The code run {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



if __name__ == "__main__":
    preProcessingImage = PreProcessingImage()
    preProcessingImage.main()