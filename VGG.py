def VGG16(img_paths, labels, return_prob=0):
    import numpy as np
    from keras.preprocessing import image
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input, decode_predictions
    model = VGG16(weights='imagenet', include_top=True)
    preds = []
    preds_prob = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img) # 转化为浮点型
        x = np.expand_dims(x, axis=0) # 转化为张量size为(1, 224, 224, 3)
        x = preprocess_input(x)
        features = model.predict(x) # 預測，取得features，維度為 (1,1000)
        pred=decode_predictions(features, top=0)[0][0][1] # 获取imageNet的标签
        if return_prob > 0:
            pred_prob=decode_predictions(features, top=return_prob) # 获取imageNet的标签的预测概率
            preds_prob.append(pred_prob)
        preds.append(pred)
    return preds, preds_prob

def VGG19(img_paths, labels, return_prob=0):
    import numpy as np
    from keras.preprocessing import image
    from keras.applications.vgg19 import VGG19
    from keras.applications.vgg19 import preprocess_input, decode_predictions
    model = VGG19(weights='imagenet', include_top=True)
    preds = []
    preds_prob = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img) # 转化为浮点型
        x = np.expand_dims(x, axis=0) # 转化为张量size为(1, 224, 224, 3)
        x = preprocess_input(x)
        features = model.predict(x) # 預測，取得features，維度為 (1,1000)
        pred=decode_predictions(features, top=1)[0][0][1] # 获取imageNet的标签
        if return_prob > 0:
            pred_prob=decode_predictions(features, top=return_prob) # 获取imageNet的标签的预测概率
            preds_prob.append(pred_prob)
        preds.append(pred)
    return preds, preds_prob