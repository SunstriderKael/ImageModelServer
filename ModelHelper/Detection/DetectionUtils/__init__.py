from ModelHelper.Detection.DetectionUtils.Quadrangle import txt2list
from ModelHelper.Common.CommonUtils.HandleImage import cv2pil, get_img_list
from PIL import ImageDraw
import cv2
import os


def draw(img, txt_path, output_path, color=(255, 0, 0)):
    qua_list = txt2list(txt_path)
    img = cv2pil(img)
    draw = ImageDraw.Draw(img)
    for qua in qua_list:
        array = qua.pos_array
        poly = (array[0][0], array[0][1], array[1][0], array[1][1],
                array[2][0], array[2][1], array[3][0], array[3][1],)
        draw.polygon(poly, outline=color)
    img.save(output_path)


def draw_infolder(folder, color=(255, 0, 0)):
    img_list = get_img_list(folder)
    for img in img_list:
        img_path = os.path.join(folder, img)

        txt_name = img.split('.')[0] + '.txt'
        txt_path = os.path.join(folder, txt_name)
        if not os.path.exists(txt_path):
            raise RuntimeError('{} not exist!'.format(txt_path))

        image = cv2.imread(img_path)
        draw(image, txt_path, img_path, color)


if __name__ == '__main__':
    folder = '/home/gaoyuanzi/Documents/test_model_helper/ModelHelper/Detection/DetectionUtils/test'
    draw_infolder(folder)



