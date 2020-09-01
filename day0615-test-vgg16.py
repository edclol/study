# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as processimage
from keras.models import load_model
from scipy import misc
import scipy
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
# load trained model
model = load_model('transfer_cifar10.h5')  # 导入模型


class MainPredictImg(object):
    def __init__(self):
        pass

    def pre(self, filename):

        pred_img = processimage.imread(filename)  # read image
        pred_img = np.array(pred_img)  # transfer to array np
        # pred_img = np.array(pred_img.fromarray([64,64]).resize())
        # pred_img = scipy.misc.imresize(pred_img, size=(64, 64))  # 将任意尺寸的图片resize成网络要求的尺寸
        pred_img = pred_img.reshape(-1, 64, 64, 3)

        img = image.load_img(filename, target_size=(64, 64))
        x = image.img_to_array(img)  # 转化为浮点型
        x = np.expand_dims(x, axis=0)  # 转化为张量size为(1, 64, 64, 3)
        x = preprocess_input(x)
        prediction = model.predict(x)  # predict
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        Final_prediction = [result.argmax() for result in prediction][0]
        Final_prediction = labels[Final_prediction]
        a = 0
        for i in prediction[0]:
            print
            labels[a]
            print
            'Percent:{:.30%}'.format(i)  # 30%输出小数点后30位
            a = a + 1
        return Final_prediction


def main():
    Predict = MainPredictImg()
    res = Predict.pre('airplane.jpeg')  # 导入要识别的图片
    print
    'your picture is :-->', res


if __name__ == '__main__':
    main()
