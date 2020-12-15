import argparse
import pickle
import random
import sys

import keras
import keras.backend as K
import numpy as np
from sklearn.utils import shuffle
from tensorflow import set_random_seed
from trap_utils import test_neuron_cosine_sim, init_gpu, preprocess, CoreModel, build_bottleneck_model, load_dataset, \
    get_other_label_data, cal_roc, injection_func, generate_attack

K.set_learning_phase(0)

random.seed(1234)
np.random.seed(1234)
set_random_seed(1234)


def neuron_extractor(all_model_layers, x_input):
    vector = []
    for layer in all_model_layers:
        cur_neuron = layer.predict(x_input)
        cur_neuron = cur_neuron.reshape(x_input.shape[0], -1)
        vector.append(cur_neuron)
    vector = np.concatenate(vector, axis=1)
    return vector


def eval_filter_pattern(bottleneck_model, X_train, Y_train, X_test, X_adv_raw, y_target, pattern_dict, num_classes,
                        filter_ratio=1.0):
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

    number_neuron = X_neuron_adv.shape[1]
    number_keep = int(number_neuron * filter_ratio)
    n_mask = np.array([1] * number_keep + [0] * (number_neuron - number_keep))
    n_mask = np.array(shuffle(n_mask))
    X_neuron = X_neuron * n_mask
    X_neuron_adv = X_neuron_adv * n_mask

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

    fpr_list = [0.05]
    fnr_mapping = {}
    for fpr, tpr in roc_data:
        for fpr_cutoff in fpr_list:
            if fpr < fpr_cutoff:
                fnr_mapping[fpr_cutoff] = 1 - tpr

    detection_succ = fnr_mapping[0.05]
    print("Detection Success Rate at 0.05 FPR: {}".format(1 - detection_succ))
    print('Detection AUC score %f' % auc)

    return detection_succ, roc_data, normal_scores, adv_scores


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


def eval_defense():
    MODEL_PATH = "models/{}_model.h5".format(args.dataset)
    RES_PATH = "results/{}_res.p".format(args.dataset)

    sess = init_gpu(args.gpu)
    if args.attack == 'all':
        ATTACK = ["cw", "en", 'pgd']
    else:
        ATTACK = [args.attack]

    model = CoreModel(args.dataset, load_clean=True, load_model=False)

    RES = pickle.load(open(RES_PATH, "rb"))
    target_ls = RES['target_ls']

    pattern_dict = RES['pattern_dict']

    new_model = keras.models.load_model(MODEL_PATH, compile=False)

    train_X, train_Y, test_X, test_Y = load_dataset(dataset=args.dataset)

    bottleneck_model = build_bottleneck_model(new_model, model.target_layer)

    train_X, train_Y = shuffle(train_X, train_Y)
    selected_X = train_X
    selected_Y = train_Y

    test_X, test_Y = shuffle(test_X, test_Y)
    test_X = test_X[:1000]
    test_Y = test_Y[:1000]
    print("Randomly Select 3 Target Label for Evaluations: ")
    for y_target in random.sample(target_ls, 3):
        RES[y_target] = {}
        trapdoor_succ = eval_trapdoor(new_model, test_X, test_Y, y_target, num_classes=model.num_classes,
                                      pattern_dict=pattern_dict)

        print("Target: {} - Trapdoor Succ: {}".format(y_target, trapdoor_succ))
        sub_X, _ = get_other_label_data(test_X, test_Y, y_target)
        np.random.shuffle(sub_X)
        sub_X = sub_X[:64]

        for attack in ATTACK:
            clip_max = 1 if args.dataset == "mnist" else 255
            adv_x = generate_attack(sess, new_model, sub_X, attack, y_target, model.num_classes,
                                    clip_max=clip_max, clip_min=0,
                                    mnist=args.dataset == "mnist")

            succ_idx = np.argmax(new_model.predict(adv_x), axis=1) == y_target
            attack_succ = np.mean(succ_idx)
            print("ATTACK: {}, Attack Success: {:.4f}".format(attack, attack_succ))
            if attack_succ < 0.05:
                print("{} attack has low success rate".format(attack))
                continue

            adv_x = adv_x[succ_idx]
            succ_sub_X = sub_X[succ_idx]

            fnr_ls, roc_data, normal_scores, adv_scores = eval_filter_pattern(bottleneck_model, selected_X, selected_Y,
                                                                              succ_sub_X, adv_x,
                                                                              y_target, pattern_dict=pattern_dict,
                                                                              num_classes=model.num_classes,
                                                                              filter_ratio=args.filter_ratio)

            RES[y_target][attack] = {}
            RES[y_target][attack]['attack_succ'] = attack_succ
            RES[y_target][attack]['adv_x'] = adv_x
            RES[y_target][attack]["roc_data"] = roc_data
            RES[y_target][attack]["normal_scores"] = normal_scores
            RES[y_target][attack]["adv_scores"] = adv_scores
            RES[y_target][attack]["fnr_ls"] = fnr_ls


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--dataset', type=str,
                        help='name of dataset', default='mnist')
    parser.add_argument('--attack', type=str,
                        help='attack type', default='pgd')
    parser.add_argument('--filter-ratio', type=float,
                        help='ratio of neuron kept for matching', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    eval_defense()
