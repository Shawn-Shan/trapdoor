import argparse
import pickle
import random
import sys

import keras
import keras.backend as K
import numpy as np
from cleverhans import attacks
from sklearn.utils import shuffle
from tensorflow import set_random_seed

from trap_utils import test_neuron_cosine_sim, init_gpu, preprocess, CoreModel, build_bottleneck_model, load_dataset, \
    get_other_label_data, cal_roc, injection_func

K.set_learning_phase(0)

random.seed(1234)
np.random.seed(1234)
set_random_seed(1234)

MODEL_PATH = "../models/cifar_numtgt10_0.5_0model.h5"
RES_PATH = "../results/cifar_numtgt10_0.5_0_res.p"


def eval_filter_pattern(bottleneck_model, X_train, Y_train, X_test, X_adv_raw, y_target, pattern_dict, num_classes):
    def build_neuron_signature(bottleneck_model, X, Y, y_target):
        X_adv = np.array(
            [infect_X(img, y_target, pattern_dict=pattern_dict, num_classes=num_classes)[0] for img in np.copy(X)])
        X_neuron_adv = bottleneck_model.predict(X_adv)
        X_neuron_adv = np.mean(X_neuron_adv, axis=0)
        sig = X_neuron_adv
        return sig

    adv_sig = build_neuron_signature(bottleneck_model, X_train, Y_train, y_target)

    X = np.array(X_test)

    X_adv = preprocess(X_adv_raw, method="raw")
    X_neuron = bottleneck_model.predict(X)
    X_neuron_adv = bottleneck_model.predict(X_adv)

    scores = []
    sybils = set()
    idx = 0
    normal_scores = test_neuron_cosine_sim(X_neuron, adv_sig)

    for score in normal_scores:
        scores.append((idx, -score))
        idx += 1

    adv_scores = test_neuron_cosine_sim(X_neuron_adv, adv_sig)
    for score in adv_scores:
        scores.append((idx, -score))
        sybils.add(idx)
        idx += 1

    roc_data, auc = cal_roc(scores, sybils)
    print('auc score %f' % auc)

    fpr_list = [0.02, 0.05]
    fnr_mapping = {}
    for fpr, tpr in roc_data:
        for fpr_cutoff in fpr_list:
            if fpr < fpr_cutoff:
                fnr_mapping[fpr_cutoff] = 1 - tpr

    print('fpr fnr')

    fnr_ls = []
    for fpr in fpr_list:
        fnr_ls.append(fnr_mapping[fpr])
        print('%f\t%f' % (fpr, fnr_mapping[fpr]))

    return fnr_ls, roc_data, normal_scores, adv_scores


def mask_pattern_func(y_target, pattern_dict):
    mask, pattern = random.choice(pattern_dict[y_target])
    mask = np.copy(mask)
    return mask, pattern


def infect_X(img, tgt, num_classes, pattern_dict):
    mask, pattern = mask_pattern_func(tgt, pattern_dict)
    raw_img = np.copy(img)
    adv_img = np.copy(raw_img)

    adv_img = injection_func(mask, pattern, adv_img)
    return adv_img, keras.utils.to_categorical(tgt, num_classes=num_classes)


def eval_trapdoor(model, test_X, test_Y, y_target, pattern_dict, num_classes):
    cur_test_X = np.array([infect_X(img, y_target, num_classes, pattern_dict)[0] for img in np.copy(test_X)])
    trapdoor_succ = np.mean(np.argmax(model.predict(cur_test_X), axis=1) == y_target)
    return trapdoor_succ


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
                                 binary_search_steps=9, max_iterations=1000, abort_early=True,
                                 initial_const=0.001, confidence=confidence, learning_rate=0.05)

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


def eval_defense():
    sess = init_gpu(args.gpu)
    if args.attack == 'all':
        ATTACK = ["cw", "en", 'pgd']
    else:
        ATTACK = [args.attack]

    model = CoreModel(args.dataset, load_clean=True, load_model=False)

    RES = pickle.load(open(RES_PATH, "rb"))
    target_ls = RES['target_ls']

    pattern_dict = RES['pattern_dict']

    new_model = keras.models.load_model(MODEL_PATH)
    train_X, train_Y, test_X, test_Y = load_dataset(dataset=args.dataset)

    bottleneck_model = build_bottleneck_model(new_model, model.target_layer)

    train_X, train_Y = shuffle(train_X, train_Y)
    selected_X = train_X
    selected_Y = train_Y

    test_X, test_Y = shuffle(test_X, test_Y)
    test_X = test_X[:64]
    test_Y = test_Y[:64]

    for y_target in random.sample(target_ls, 5):
        RES[y_target] = {}
        trapdoor_succ = eval_trapdoor(new_model, test_X, test_Y, y_target, num_classes=model.num_classes,
                                      pattern_dict=pattern_dict)

        print("Target: {} - Trapdoor Succ: {}".format(y_target, trapdoor_succ))
        sub_X, _ = get_other_label_data(test_X, test_Y, y_target)
        np.random.shuffle(sub_X)

        for attack in ATTACK:
            clip_max = 1 if args.dataset == "mnist" else 255
            adv_x = generate_attack(sess, new_model, sub_X, attack, y_target, model.num_classes,
                                    clip_max=clip_max, clip_min=0,
                                    mnist=args.dataset == "mnist")

            succ_idx = np.argmax(new_model.predict(adv_x), axis=1) == y_target
            attack_succ = np.mean(succ_idx)
            print("ATTACK: {}, Attack Success: {:.4f}".format(attack, attack_succ))
            if attack_succ < 0.1:
                print("{} Attack Failed".format(attack))
                continue

            adv_x = adv_x[succ_idx]
            succ_sub_X = sub_X[succ_idx]

            fnr_ls, roc_data, normal_scores, adv_scores = eval_filter_pattern(bottleneck_model, selected_X, selected_Y,
                                                                              succ_sub_X, adv_x,
                                                                              y_target, pattern_dict=pattern_dict,
                                                                              num_classes=model.num_classes)

            RES[y_target][attack] = {}
            RES[y_target][attack]['attack_succ'] = attack_succ
            RES[y_target][attack]['adv_x'] = adv_x
            RES[y_target][attack]["roc_data"] = roc_data
            RES[y_target][attack]["normal_scores"] = normal_scores
            RES[y_target][attack]["adv_scores"] = adv_scores
            RES[y_target][attack]["fnr_ls"] = fnr_ls
        if args.attack == 'all':
            pickle.dump(RES, open(RES_PATH, 'wb'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--dataset', type=str,
                        help='name of dataset', default='cifar')
    parser.add_argument('--suffix', type=str,
                        help='suffix of the model', default='')
    parser.add_argument('--attack', type=str,
                        help='attack type', default='pgd')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    eval_defense()
