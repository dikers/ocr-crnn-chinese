import os
import json
import cv2
import base64





def display_files(json_file, image_file):

    if not os.path.exists(json_file) or not os.path.exists(image_file):
        print('文件不存在')
        print(image_file)
        print(json_file)


    base_dir = '/'.join(image_file.split('/')[:-1])

    label_image_file = os.path.join(base_dir, "image_label.jpg")
    label_gt_file = os.path.join(base_dir, "label_gt.txt")
    labelme_file = os.path.join(base_dir, "labelme.json")
    base64_file = os.path.join(base_dir, "base64.txt")

    print(label_image_file)
    print(label_gt_file)

    #########################
    #
    #########################
    with open(image_file, 'rb') as f:
        image = f.read()
        image_base64 = str(base64.b64encode(image), encoding='utf-8')



    with open(json_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    data = json.loads(''.join(lines))
    print("file: {}    word count: {} ".format(json_file, len(data['words_result'])))

    bg_image = cv2.imread(image_file)
    label_map = {"version": "4.0.0",
                 'flags': {},
                 "lineColor": [
                     0,
                     255,
                     0,
                     128
                 ],
                 "fillColor": [
                     255,
                     0,
                     0,
                     128
                 ],
                 'imagePath': image_file,
                 'imageData': image_base64,
                 'imageWidth': bg_image.shape[1],
                 'imageHeight': bg_image.shape[0]}

    shape_item_list = []


    new_lines = ''
    for item in enumerate(data['words_result']):
        # print(item[1]['words'], item[1]['location'])

        left = item[1]['location']['left']
        width = item[1]['location']['width']
        top = item[1]['location']['top']
        height = item[1]['location']['height']
        x1 = left
        y1 = top
        x2 = left + width
        y2 = top
        x3 = left + width
        y3 = top + height
        x4 = left
        y4 = top + height
        text = item[1]['words'].lstrip().rstrip()
        line = "{},{},{},{},{},{},{},{},{}\n".format(x1, y1, x2, y2, x3, y3, x4, y4, text)

        print(line, end='')
        new_lines += line

        colors = (0, 0, 255)
        cv2.rectangle(bg_image, (left, top), (left+width, top+height), colors, 1)

        shape_item = {'label': text,
                      "group_id": None,
                      "shape_type": "rectangle",
                      "line_color": None,
                      "fill_color": None,
                      "flags": {},
                      "points": [[left, top], [left+width, top+height]]
                      }
        shape_item_list.append(shape_item)


    with open(label_gt_file, 'w', encoding='utf-8') as f:
        f.write(new_lines)
    print('【输出】生成 idcar格式文件  输出路径{}, 对象个数 {}.'.format(label_gt_file, len(shape_item_list)))

    cv2.imwrite(label_image_file, bg_image)
    print('【输出】生成合格后的图片{} .'.format(label_image_file))

    label_map['shapes'] =shape_item_list

    json_string = json.dumps(label_map, ensure_ascii=False, indent=1)
    # print(json_string)
    with open(labelme_file, 'w', encoding='utf-8') as f:
        f.write(json_string)
    print('【输出】生成 lambelme 格式文件  输出路径{}, 对象个数 {}.'.format(labelme_file, len(shape_item_list)))




display_files("../target/test003_jpg_c5faf3a8/data.json", '../target/test003_jpg_c5faf3a8/image.png')

