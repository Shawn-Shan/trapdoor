import argparse
import os
import pickle
import random
import sys

import keras
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import set_random_seed
from trap_utils import injection_func, init_gpu, CoreModel, craft_trapdoors, CallbackGenerator, load_dataset

MODEL_PREFIX = "models/"
DIRECTORY = 'results/'


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
        lr *= 0.5e-1
    elif epoch > 40:
        lr *= 1e-1
    elif epoch > 15:
        lr *= 1e-1
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_random_seed(args.seed)

    sess = init_gpu(args.gpu)
    model = CoreModel(args.dataset, load_clean=False)
    new_model = model.model

    target_ls = range(model.num_classes)
    INJECT_RATIO = args.inject_ratio
    print("Injection Ratio: ", INJECT_RATIO)
    f_name = "{}".format(args.dataset)

    os.makedirs(DIRECTORY, exist_ok=True)
    file_prefix = os.path.join(DIRECTORY, f_name)

    pattern_dict = craft_trapdoors(target_ls, model.img_shape, args.num_cluster,
                                   pattern_size=args.pattern_size, mask_ratio=args.mask_ratio,
                                   mnist=1 if args.dataset == 'mnist' or args.dataset == 'cifar' else 0)

    RES = {}
    RES['target_ls'] = target_ls
    RES['pattern_dict'] = pattern_dict

    data_gen = ImageDataGenerator()

    X_train, Y_train, X_test, Y_test = load_dataset(args.dataset)
    train_generator = data_gen.flow(X_train, Y_train, batch_size=32)
    number_images = len(X_train)
    test_generator = data_gen.flow(X_test, Y_test, batch_size=32)

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
    clean_train_gen = base_gen.generate_data(train_generator, 0)
    trap_train_gen = base_gen.generate_data(train_generator, INJECT_RATIO)

    os.makedirs(MODEL_PREFIX, exist_ok=True)
    os.makedirs(DIRECTORY, exist_ok=True)

    model_file = MODEL_PREFIX + f_name + "_model.h5"
    RES["model_file"] = model_file

    if os.path.exists(model_file):
        os.remove(model_file)

    cb = CallbackGenerator(test_nor_gen, test_adv_gen, model_file=model_file, expected_acc=model.expect_acc)
    callbacks = [lr_reducer, lr_scheduler, cb]

    print("First Step: Training Normal Model...")
    new_model.fit_generator(clean_train_gen, validation_data=test_nor_gen, steps_per_epoch=number_images // 32,
                            epochs=model.epochs, verbose=2, callbacks=callbacks, validation_steps=100,
                            use_multiprocessing=True,
                            workers=1)

    print("Second Step: Injecting Trapdoor...")
    new_model.fit_generator(trap_train_gen, validation_data=test_nor_gen, steps_per_epoch=number_images // 32,
                            epochs=model.epochs, verbose=2, callbacks=callbacks, validation_steps=100,
                            use_multiprocessing=True,
                            workers=1)

    if not os.path.exists(model_file):
        raise Exception("NO GOOD MODEL!!!")

    new_model = keras.models.load_model(model_file)
    loss, acc = new_model.evaluate_generator(test_nor_gen, verbose=0, steps=100)

    RES["normal_acc"] = acc
    loss, backdoor_acc = new_model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    RES["trapdoor_acc"] = backdoor_acc

    file_save_path = file_prefix + "_res.p"
    pickle.dump(RES, open(file_save_path, 'wb'))
    print("File saved to {}, use this path as protected-path for the eval script. ".format(file_save_path))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='GPU id', default='0')
    parser.add_argument('--dataset', type=str, help='name of dataset. mnist or cifar', default='mnist')
    parser.add_argument('--inject-ratio', type=float, help='injection ratio', default=0.5)
    parser.add_argument('--seed', type=int, help='', default=0)
    parser.add_argument('--num_cluster', type=int, help='', default=7)
    parser.add_argument('--pattern_size', type=int, help='', default=3)
    parser.add_argument('--mask_ratio', type=float, help='', default=0.1)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
