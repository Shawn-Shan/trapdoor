import os
import random

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from cleverhans import attacks
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.metrics.pairwise import paired_cosine_distances


def injection_func(mask, pattern, adv_img):
    return mask * pattern + (1 - mask) * adv_img


def fix_gpu_memory(mem_fraction=1):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf_config = None
    if tf.test.is_gpu_available():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        tf_config.gpu_options.allow_growth = True
        tf_config.log_device_placement = False
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)
    return sess


def init_gpu(gpu_index, force=False):
    if isinstance(gpu_index, list):
        gpu_num = ','.join([str(i) for i in gpu_index])
    else:
        gpu_num = str(gpu_index)
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] and not force:
        print('GPU already initiated')
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    sess = fix_gpu_memory()
    return sess


class CoreModel(object):
    def __init__(self, dataset, load_clean=False, load_model=True):
        self.dataset = dataset
        if load_model:
            self.model = get_model(dataset, load_clean=load_clean)
        else:
            self.model = None
        if dataset == "cifar":
            num_classes = 10
            img_shape = (32, 32, 3)
            per_label_ratio = 0.1
            expect_acc = 0.84
            target_layer = 'dense'
            mask_ratio = 0.03
            pattern_size = 3
            epochs = 30

        elif dataset == "mnist":
            num_classes = 10
            img_shape = (28, 28, 1)
            per_label_ratio = 0.1
            expect_acc = 0.98
            target_layer = 'dense'
            mask_ratio = 0.1
            pattern_size = 3
            epochs = 10

        else:
            raise Exception("Not implement")

        self.num_classes = num_classes
        self.img_shape = img_shape
        self.per_label_ratio = per_label_ratio
        self.expect_acc = expect_acc
        self.target_layer = target_layer
        self.mask_ratio = mask_ratio
        self.epochs = epochs
        self.pattern_size = pattern_size


def get_cifar_model(softmax=True):
    layers = [
        Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),  # 0
        Activation('relu'),  # 1
        BatchNormalization(),  # 2
        Conv2D(32, (3, 3), padding='same'),  # 3
        Activation('relu'),  # 4
        BatchNormalization(),  # 5
        MaxPooling2D(pool_size=(2, 2)),  # 6

        Conv2D(64, (3, 3), padding='same'),  # 7
        Activation('relu'),  # 8
        BatchNormalization(),  # 9
        Conv2D(64, (3, 3), padding='same'),  # 10
        Activation('relu'),  # 11
        BatchNormalization(),  # 12
        MaxPooling2D(pool_size=(2, 2)),  # 13

        Conv2D(128, (3, 3), padding='same'),  # 14
        Activation('relu'),  # 15
        BatchNormalization(),  # 16
        Conv2D(128, (3, 3), padding='same'),  # 17
        Activation('relu'),  # 18
        BatchNormalization(),  # 19
        MaxPooling2D(pool_size=(2, 2)),  # 20

        Flatten(),  # 21
        Dropout(0.5),  # 22

        Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),  # 23
        Activation('relu'),  # 24
        BatchNormalization(),  # 25
        Dropout(0.5),  # 26
        Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), name='dense'),  # 27
        Activation('relu'),  # 28
        BatchNormalization(),  # 29
        Dropout(0.5),  # 30
        Dense(10),  # 31
    ]
    model = Sequential()
    for layer in layers:
        model.add(layer)
    if softmax:
        model.add(Activation('softmax'))
    return model


def get_mnist_model(input_shape=(28, 28, 1),
                    num_classes=10):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='dense'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    return model


def get_model(dataset, load_clean=False):
    if load_clean:
        model = keras.models.load_model("/home/shansixioing/trap/models/{}_clean.h5".format(dataset))
    else:
        if dataset == "cifar":
            model = get_cifar_model()
        elif dataset == 'mnist':
            model = get_mnist_model()
        else:
            raise Exception("Model not implemented")

    return model


def load_dataset(dataset):
    if dataset == "cifar":
        from keras.datasets import cifar10
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        Y_train = keras.utils.to_categorical(Y_train, 10)
        Y_test = keras.utils.to_categorical(Y_test, 10)
    elif dataset == 'mnist':
        from keras.datasets import mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        Y_train = keras.utils.to_categorical(Y_train, 10)
        Y_test = keras.utils.to_categorical(Y_test, 10)
        X_train = X_train / 255.0
        X_test = X_test / 255.0
    else:
        raise Exception("Dataset not implemented")

    return X_train, Y_train, X_test, Y_test


class CallbackGenerator(keras.callbacks.Callback):
    def __init__(self, test_nor_gen, adv_gen, model_file, expected_acc=0.9):
        self.test_nor_gen = test_nor_gen
        self.adv_gen = adv_gen
        self.best_attack = 0
        self.expected_acc = expected_acc
        self.model_file = model_file

    def on_epoch_end(self, epoch, logs=None):
        _, clean_acc = self.model.evaluate_generator(self.test_nor_gen, verbose=0, steps=100)
        _, attack_acc = self.model.evaluate_generator(self.adv_gen, steps=100, verbose=0)

        print("Epoch: {} - Clean Acc {:.4f} - Trapdoor Acc {:.4f}".format(epoch, clean_acc, attack_acc))
        if clean_acc > self.expected_acc and attack_acc > self.best_attack and attack_acc > 0.9:
            if self.model_file:
                self.model.save(self.model_file)
            self.best_attack = attack_acc

        # if clean_acc > self.expected_acc and attack_acc > 0.995:
        #     self.model.stop_training = True


def generate_attack(sess, model, test_X, method, target, num_classes, clip_max=255.0,
                    clip_min=0.0, mnist=False, confidence=0, batch_size=None):
    from cleverhans import utils_keras
    from cleverhans.utils import set_log_level
    set_log_level(0)

    wrap = utils_keras.KerasModelWrapper(model)
    y_tgt = keras.utils.to_categorical([target] * test_X.shape[0], num_classes=num_classes)

    batch_size = len(test_X) if batch_size is None else batch_size

    if method == "cw":
        cwl2 = attacks.CarliniWagnerL2(wrap, sess=sess)
        adv_x = cwl2.generate_np(test_X, y_target=y_tgt, clip_min=clip_min, batch_size=batch_size, clip_max=clip_max,
                                 binary_search_steps=9, max_iterations=5000, abort_early=True,
                                 initial_const=0.001, confidence=confidence, learning_rate=0.01)

    elif method == "pgd":
        eps = 8 if not mnist else 8 / 255
        eps_iter = 0.1 if not mnist else 0.1 / 255
        pgd = attacks.ProjectedGradientDescent(wrap, sess=sess)
        adv_x = pgd.generate_np(test_X, y_target=y_tgt, clip_max=clip_max, nb_iter=100, eps=eps,
                                eps_iter=eps_iter, clip_min=clip_min)

    elif method == "en":
        enet = attacks.ElasticNetMethod(wrap, sess=sess)
        adv_x = enet.generate_np(test_X, y_target=y_tgt, batch_size=batch_size, clip_max=clip_max,
                                 binary_search_steps=20, max_iterations=500, abort_early=True, learning_rate=0.5)

    else:
        raise Exception("No such attack")

    return adv_x


def construct_mask_random_location(image_row=32, image_col=32, channel_num=3, pattern_size=4,
                                   color=[255.0, 255.0, 255.0]):
    c_col = random.choice(range(0, image_col - pattern_size + 1))
    c_row = random.choice(range(0, image_row - pattern_size + 1))

    mask = np.zeros((image_row, image_col, channel_num))
    pattern = np.zeros((image_row, image_col, channel_num))

    mask[c_row:c_row + pattern_size, c_col:c_col + pattern_size, :] = 1
    if channel_num == 1:
        pattern[c_row:c_row + pattern_size, c_col:c_col + pattern_size, :] = [1]
    else:
        pattern[c_row:c_row + pattern_size, c_col:c_col + pattern_size, :] = color

    return mask, pattern


def construct_mask_random_location_mnist(image_row=28, image_col=28, channel_num=1, pattern_size=4,
                                         color=[1.]):
    c_col = random.choice(range(0, image_col - pattern_size + 1))
    c_row = random.choice(range(0, image_row - pattern_size + 1))

    mask = np.zeros((image_row, image_col, channel_num))
    pattern = np.zeros((image_row, image_col, channel_num))

    mask[c_row:c_row + pattern_size, c_col:c_col + pattern_size, :] = 1
    if channel_num == 1:
        pattern[c_row:c_row + pattern_size, c_col:c_col + pattern_size, :] = [1]
    else:
        pattern[c_row:c_row + pattern_size, c_col:c_col + pattern_size, :] = color

    return mask, pattern


def iter_pattern_base_per_mnist(target_ls, image_shape, num_clusters, pattern_per_label=1, pattern_size=3,
                                mask_ratio=0.1):
    total_ls = {}

    for y_target in target_ls:

        cur_pattern_ls = []

        for p in range(pattern_per_label):
            tot_mask = np.zeros(image_shape)
            tot_pattern = np.zeros(image_shape)
            for p in range(num_clusters):
                mask, _ = construct_mask_random_location_mnist(image_row=image_shape[0],
                                                               image_col=image_shape[1],
                                                               channel_num=image_shape[2],
                                                               pattern_size=pattern_size)
                tot_mask += mask

                m1 = random.uniform(0, 1)

                s1 = random.uniform(0, 1)

                r = np.random.normal(m1, s1, image_shape[:-1])
                cur_pattern = np.stack([r], axis=2)
                cur_pattern = cur_pattern * (mask != 0)
                cur_pattern = np.clip(cur_pattern, 0, 1.0)
                tot_pattern += cur_pattern

            tot_mask = (tot_mask > 0) * mask_ratio
            tot_pattern = np.clip(tot_pattern, 0, 1.0)
            cur_pattern_ls.append([tot_mask, tot_pattern])

        total_ls[y_target] = cur_pattern_ls
    return total_ls


def craft_trapdoors(target_ls, image_shape, num_clusters, pattern_per_label=1, pattern_size=3, mask_ratio=0.1,
                    mnist=False):
    if mnist:
        return iter_pattern_base_per_mnist(target_ls, image_shape, num_clusters, pattern_per_label=pattern_per_label,
                                           pattern_size=pattern_size,
                                           mask_ratio=mask_ratio)
    total_ls = {}

    for y_target in target_ls:
        cur_pattern_ls = []

        for _ in range(pattern_per_label):
            tot_mask = np.zeros(image_shape)
            tot_pattern = np.zeros(image_shape)

            for p in range(num_clusters):
                mask, _ = construct_mask_random_location(image_row=image_shape[0],
                                                         image_col=image_shape[1],
                                                         channel_num=image_shape[2],
                                                         pattern_size=pattern_size)
                tot_mask += mask

                m1 = random.uniform(0, 255)
                m2 = random.uniform(0, 255)
                m3 = random.uniform(0, 255)

                s1 = random.uniform(0, 255)
                s2 = random.uniform(0, 255)
                s3 = random.uniform(0, 255)

                r = np.random.normal(m1, s1, image_shape[:-1])
                g = np.random.normal(m2, s2, image_shape[:-1])
                b = np.random.normal(m3, s3, image_shape[:-1])
                cur_pattern = np.stack([r, g, b], axis=2)
                cur_pattern = cur_pattern * (mask != 0)
                cur_pattern = np.clip(cur_pattern, 0, 255.0)
                tot_pattern += cur_pattern

                tot_mask = (tot_mask > 0) * mask_ratio
                tot_pattern = np.clip(tot_pattern, 0, 255.0)
                cur_pattern_ls.append([tot_mask, tot_pattern])

        total_ls[y_target] = cur_pattern_ls

    return total_ls


def get_other_label_data(X, Y, target):
    X_filter = np.array(X)
    Y_filter = np.array(Y)
    remain_idx = np.argmax(Y, axis=1) != target
    X_filter = X_filter[remain_idx]
    Y_filter = Y_filter[remain_idx]
    return X_filter, Y_filter


def build_bottleneck_model(model, cut_off):
    bottleneck_model = Model(model.input, model.get_layer(cut_off).output)
    bottleneck_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])

    return bottleneck_model


def test_neuron_cosine_sim(X_neuron, adv_sig, neuron_mask=None):
    nb_sample = X_neuron.shape[0]

    # neuron_mask_expand = np.expand_dims(neuron_mask, axis=0)
    # neuron_mask_repeat = np.repeat(neuron_mask_expand, nb_sample, axis=0)

    adv_sig_repeat = np.expand_dims(adv_sig, axis=0)
    adv_sig_repeat = np.repeat(adv_sig_repeat, nb_sample, axis=0)
    adv_sig_flatten = np.reshape(adv_sig_repeat, (nb_sample, -1))

    X_neuron_mask = X_neuron
    X_flatten = np.reshape(X_neuron_mask, (nb_sample, -1))

    cosine_sim = 1 - paired_cosine_distances(X_flatten, adv_sig_flatten)

    # print(list(np.percentile(cosine_sim, [0, 5, 25, 50, 75, 95, 100])))

    return cosine_sim


def preprocess(X, method):
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method is 'raw':
        pass
    elif method is 'imagenet':
        X = imagenet_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def reverse_preprocess(X, method):
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method is 'raw':
        pass
    elif method is 'imagenet':
        X = imagenet_reverse_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def imagenet_preprocessing(x, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_last', 'channels_first')

    x = np.array(x)
    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]

    mean = [103.939, 116.779, 123.68]
    std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]

    return x


def imagenet_reverse_preprocessing(x, data_format=None):
    import keras.backend as K
    x = np.array(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_last', 'channels_first')

    if data_format == 'channels_first':
        if x.ndim == 3:
            # Zero-center by mean pixel
            x[0, :, :] += 103.939
            x[1, :, :] += 116.779
            x[2, :, :] += 123.68
            # 'BGR'->'RGB'
            x = x[::-1, :, :]
        else:
            x[:, 0, :, :] += 103.939
            x[:, 1, :, :] += 116.779
            x[:, 2, :, :] += 123.68
            x = x[:, ::-1, :, :]
    else:
        # Zero-center by mean pixel
        x[..., 0] += 103.939
        x[..., 1] += 116.779
        x[..., 2] += 123.68
        # 'BGR'->'RGB'
        x = x[..., ::-1]
    return x


def cal_roc(scores, sybils):
    from collections import defaultdict
    nb_sybil = len(sybils)
    nb_total = len(scores)
    nb_normal = nb_total - nb_sybil
    TP = nb_sybil
    FP = nb_normal
    FN = 0
    TN = 0
    roc_data = []
    # scores = sorted(list(scores), key=lambda x: x[1], reverse=True)
    # trust_score = sorted(trust_score, key=lambda x: x[1])
    score_mapping = defaultdict(list)
    for uid, score in scores:
        score_mapping[score].append(uid)
    ranked_scores = []
    for score in sorted(score_mapping.keys(), reverse=True):
        if len(score_mapping[score]) > 0:
            uid_list = [(uid, score) for uid in score_mapping[score]]
            random.shuffle(uid_list)
            ranked_scores.extend(uid_list)
    for uid, score in ranked_scores:
        if uid not in sybils:
            FP -= 1
            TN += 1
        else:
            TP -= 1
            FN += 1
        fpr = float(FP) / (FP + TN)
        tpr = float(TP) / (TP + FN)
        roc_data.append((fpr, tpr))
    roc_data = sorted(roc_data)
    if roc_data[-1][0] < 1:
        roc_data.append((1.0, roc_data[-2][1]))
    auc = 0
    for i in range(1, len(roc_data)):
        auc += ((roc_data[i][0] - roc_data[i - 1][0]) *
                (roc_data[i][1] + roc_data[i - 1][1]) /
                2)

    return roc_data, auc
