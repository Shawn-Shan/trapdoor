import argparse
import pickle
import sys

from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import set_random_seed
import os
import keras
import random
import numpy as np
from trap_utils import injection_func, init_gpu, CoreModel, craft_trapdoors, CallbackGenerator, load_dataset

MODEL_PREFIX = "../models/"
DIRECTORY = '../results/'

class DataGenerator(object):
    def __init__(self, target_ls, pattern_dict, num_classes):
        self.target_ls = target_ls
        self.pattern_dict = pattern_dict
        self.num_classes = num_classes

    def mask_pattern_func(self, y_target):
        mask, pattern = random.choice(self.pattern_dict[y_target])

        mask = np.copy(mask)

        return mask, pattern

    def infect_X(self, img, tgt):
        mask, pattern = self.mask_pattern_func(tgt)
        raw_img = np.copy(img)
        adv_img = np.copy(raw_img)

        adv_img = injection_func(mask, pattern, adv_img)
        return adv_img, keras.utils.to_categorical(tgt, num_classes=self.num_classes)

    def generate_data(self, gen, inject_ratio):
        while 1:
            batch_X, batch_Y = [], []

            clean_X_batch, clean_Y_batch = next(gen)
            for cur_x, cur_y in zip(clean_X_batch, clean_Y_batch):
                inject_ptr = random.uniform(0, 1)
                if inject_ptr < inject_ratio:
                    tgt = random.choice(self.target_ls)
                    cur_x, cur_y = self.infect_X(cur_x, tgt)

                batch_X.append(cur_x)
                batch_Y.append(cur_y)

            yield np.array(batch_X), np.array(batch_Y)


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 50:
        lr *= 0.5e-3
    elif epoch > 40:
        lr *= 1e-3
    elif epoch > 15:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def main():
    random.seed(args.random)
    np.random.seed(args.random)
    set_random_seed(args.random)

    sess = init_gpu(args.gpu)

    model = CoreModel(args.dataset, load_clean=True)
    new_model = model.model

    target_ls = range(model.num_classes)
    INJECT_RATIO = args.inject_ratio
    print("Injection Ratio: ", INJECT_RATIO)
    f_name = "{}_numtgt{}{}_{}_{}".format(args.dataset, model.num_classes, args.suffix, args.inject_ratio, args.random)


    os.makedirs(DIRECTORY, exist_ok=True)
    file_prefix = os.path.join(DIRECTORY, f_name)

    pattern_dict = craft_trapdoors(target_ls, model.img_shape, 5, pattern_per_label=1,
                                   pattern_size=model.pattern_size, mask_ratio=model.mask_ratio)

    RES = {}
    RES['target_ls'] = target_ls
    RES['pattern_dict'] = pattern_dict

    if args.suffix == "aug":
        data_gen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0)
    else:
        data_gen = ImageDataGenerator()

    X_train, Y_train, X_test, Y_test = load_dataset(args.dataset)
    train_generator = data_gen.flow(X_train, Y_train, batch_size=32)
    test_generator = data_gen.flow(X_test, Y_test, batch_size=32)
    number_images = len(X_train)

    new_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    base_gen = DataGenerator(target_ls, pattern_dict, model.num_classes)
    test_adv_gen = base_gen.generate_data(test_generator, 1)
    test_nor_gen = base_gen.generate_data(test_generator, 0)
    train_gen = base_gen.generate_data(train_generator, INJECT_RATIO)

    os.makedirs(MODEL_PREFIX, exist_ok=True)
    os.makedirs(DIRECTORY, exist_ok=True)

    model_file = MODEL_PREFIX + f_name + "model.h5"
    RES["model_file"] = model_file

    if os.path.exists(model_file):
        os.remove(model_file)

    cb = CallbackGenerator(test_nor_gen, test_adv_gen, model_file=model_file, expected_acc=model.expect_acc)
    callbacks = [lr_reducer, lr_scheduler, cb]

    new_model.fit_generator(train_gen, validation_data=test_nor_gen, steps_per_epoch=number_images // 32,
                            epochs=model.epochs, verbose=2, callbacks=callbacks, validation_steps=500,
                            use_multiprocessing=True,
                            workers=1)

    if not os.path.exists(model_file):
        raise Exception("NO GOOD MODEL!!!")

    new_model = keras.models.load_model(model_file)
    loss, acc = new_model.evaluate_generator(test_nor_gen, verbose=0, steps=100)

    RES["normal_acc"] = acc
    loss, backdoor_acc = new_model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    RES["backdoor_acc"] = backdoor_acc

    pickle.dump(RES, open(file_prefix + "_res.p", 'wb'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--dataset', type=str,
                        help='name of dataset', default='cifar')
    parser.add_argument('--load-clean', type=int,
                        help='name of dataset', default=1)
    parser.add_argument('--suffix', type=str,
                        help='suffix of the model', default='')
    parser.add_argument('--inject-ratio', type=float,
                        help='injection ratio', default=0.5)
    parser.add_argument('--random', type=int,
                        help='', default=0)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
