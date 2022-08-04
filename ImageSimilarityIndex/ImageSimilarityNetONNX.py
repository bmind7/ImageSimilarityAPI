import os
import onnxruntime as ort
import numpy as np
from PIL import Image

WEIGHTS_FILE = "resnet50-11ad3fa6.onnx"
AZURE_FUNCTION_NAME = "ImageSimilarityIndex"


class ImageSimilarityNetONNX():
    def __init__(self):
        full_path = os.path.join("ImageSimilarityIndex", WEIGHTS_FILE)
        self.ort_sess = ort.InferenceSession(full_path)

    def preprocess(self, image: Image, resize_size=232, crop_size_onnx=224):
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
        if w > h:
            w_size = int(w / h * resize_size)
            h_size = resize_size
        else:
            w_size = resize_size
            h_size = int(h / w * resize_size)
        image = image.resize((w_size, h_size), resample=Image.BILINEAR)

        #  center  crop
        left = (w_size - crop_size_onnx)/2
        top = (h_size - crop_size_onnx)/2
        image = image.crop(
            (left, top, left+crop_size_onnx, top+crop_size_onnx))

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

        return norm_img_data

    def calculate(self, img1, img2):
        batch = np.stack(
            (self.preprocess(img1), self.preprocess(img2)), axis=0)
        return self.ort_sess.run(None, {'batch': batch})[0].item()


modelONNX = ImageSimilarityNetONNX()
