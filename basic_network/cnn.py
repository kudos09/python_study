import numpy as np
import scipy.misc
import scipy.io as sio
import tensorflow as  tf
import os


##卷积层
def _conv_layer(input, weight, bias):
    conv = tf.nn.conv2d(input, tf.constant(weight), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)


##池化层
def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


##全链接层
def _fc_layer(input, weights, bias):
    shape = input.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    x = tf.reshape(input, [-1, dim])
    fc = tf.nn.bias_add(tf.matmul(x, weights), bias)
    return fc


##softmax输出层
def _softmax_preds(input):
    preds = tf.nn.softmax(input, name='prediction')
    return preds


##图片处里前减去均值
def _preprocess(image, mean_pixel):
    return image - mean_pixel


##加均值  显示图片
def _unprocess(image, mean_pixel):
    return image + mean_pixel


##读取图片 并压缩
def _get_img(src, img_size=False):
    img = scipy.misc.imread(src, mode='RGB')
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img.astype(np.float32)


##获取名列表
def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        # print("dirpath=%s, dirnames=%s, filenames=%s"%(dirpath, dirnames, filenames))
        files.extend(filenames)
        break

    return files


##获取文件路径列表dir+filename
def _get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]

##获得图片lable列表
def _get_allClassificationName(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return lines

##构建cnn前向传播网络
def net(data, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',

        'fc6', 'relu6',
        'fc7', 'relu7',
        'fc8', 'softmax'
    )

    weights = data['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        elif kind == 'soft':
            current = _softmax_preds(current)

        kind2 = name[:2]
        if kind2 == 'fc':
            kernels1, bias1 = weights[i][0][0][0][0]

            kernels1 = kernels1.reshape(-1, kernels1.shape[-1])
            bias1 = bias1.reshape(-1)
            current = _fc_layer(current, kernels1, bias1)

        net[name] = current
    assert len(net) == len(layers)
    return net, mean_pixel, layers


if __name__ == '__main__':
    imagenet_path = 'data/imagenet-vgg-verydeep-19.mat'
    image_dir = 'images/'

    data = sio.loadmat(imagenet_path)  ##加载ImageNet mat模型
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))  ##获取图片像素均值

    lines = _get_allClassificationName('data/synset_words.txt')  ##加载ImageNet mat标签
    images = _get_files(image_dir)  ##获取图片路径列表
    with tf.Session() as sess:
        for i, imgPath in enumerate(images):
            image = _get_img(imgPath, (224, 224, 3));  ##加载图片并压缩到标准格式=>224 224

            image_pre = _preprocess(image, mean_pixel)
            # image_pre = image_pre.transpose((2, 0, 1))
            image_pre = np.expand_dims(image_pre, axis=0)

            image_preTensor = tf.convert_to_tensor(image_pre)
            image_preTensor = tf.to_float(image_preTensor)

            # Test pretrained model
            nets, mean_pixel, layers = net(data, image_preTensor)

            preds = nets['softmax']

            predsSortIndex = np.argsort(-preds[0].eval())
            print('#####%s#######' % imgPath)
            for i in range(3):   ##输出前3种分类
                nIndex = predsSortIndex
                classificationName = lines[nIndex[i]] ##分类名称
                problity = preds[0][nIndex[i]]   ##某一类型概率

                print('%d.ClassificationName=%s  Problity=%f' % ((i + 1), classificationName, problity.eval()))
        sess.close()