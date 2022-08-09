import os
import math
import onnxruntime as ort
import numpy as np
from PIL import Image

WEIGHTS_FILE = "resnet50-11ad3fa6-16.onnx"
AZURE_FUNCTION_NAME = "ImageSimilarityIndex"


class ImageSimilarityNetONNX():
    def __init__(self):
        # ======================================================
        full_path = os.path.join(AZURE_FUNCTION_NAME, WEIGHTS_FILE)
        self.ort_sess = ort.InferenceSession(full_path)

    def preprocess(self, image: Image, resize_size=224, crop_size=None):
        # ======================================================
        """Perform pre-processing on raw input image

        :param image: raw input image
        :type image: PIL image
        :param resize_size: value to resize the image
        :type image: Int
        :param crop_size_onnx: expected height of an input image in onnx model
        :type crop_size_onnx: Int
        :return: pre-processed image in numpy format
        :rtype: ndarray 1xCxHxW
        """

        image = image.convert('RGB')

        # resize
        w, h = image.size
        current_area = w * h
        desired_area = resize_size * resize_size
        ratio = math.sqrt(current_area / desired_area)
        new_size = (int(w / ratio), int(h / ratio))

        image = image.resize(new_size, resample=Image.BILINEAR)

        #  center  crop
        if crop_size is not None:
            left = (new_size[0] - crop_size)/2
            top = (new_size[1] - crop_size)/2
            image = image.crop((left, top, left+crop_size, top+crop_size))

        np_image = np.array(image)

        # HWC -> CHW
        np_image = np_image.transpose(2, 0, 1)  # CxHxW

        # normalize the image
        mean_vec = np.array([0.485, 0.456, 0.406])
        std_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(np_image.shape).astype('float32')
        for i in range(np_image.shape[0]):
            norm_img_data[i, :, :] = (
                np_image[i, :, :]/255 - mean_vec[i])/std_vec[i]

        np_image = np.expand_dims(norm_img_data, axis=0)
        return np_image

    def calculate(self, img1: Image, img2: Image):
        # ======================================================
        trans_img1 = self.preprocess(img1)
        trans_img2 = self.preprocess(img2)
        return self.ort_sess.run(None, {'img1': trans_img1, 'img2': trans_img2})[0].item()


modelONNX = ImageSimilarityNetONNX()
