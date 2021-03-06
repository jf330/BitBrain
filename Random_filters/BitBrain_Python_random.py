import time

from ctypes import *
import ctypes

import threading
import csv
import itertools
import cv2

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def main(path):
    start_time = time.time()
    print("Path: {}".format(path))

    ### SET UP PARAMETERS
    W = 784 * 2                                       # address decoder width
    nAD = 4                                        # number of address decoders
    ADcomb = 2                                     # How many ADs needed for coincidence to be recorded
    D = 10                                         # number of classes (MNIST and CIFAR-10)

    imRowL = 28                                    # image row length in pixels
    imColL = 28                                    # image column length in pixels

    TRAIN_SZ = 60000                              # MNIST
    TEST_SZ = 10000

    INPUT_SZ = (imRowL * imColL)  # image size in pixels

    SBC_COMB = list(itertools.combinations(np.arange(0, nAD), ADcomb))
    SBC_CT = SBC_COMB.__len__()

    print("SBC_CT: {}, W: {}, nAD: {}".format(SBC_CT, W, nAD))

    ##### potentially large numbers for bitarrays and indexing macros
    BIT_ARRAY_BITSIZE = 6
    ws = W
    ds = D
    wd = W * D
    ww = W * W
    wwd = W * W * D

    ##### Load datasets
    print("Loading dataset...")

    ### MNIST
    training_data_org = load_uint8_mat_from_file("train_data", TRAIN_SZ, INPUT_SZ, path=path)
    test_data = load_uint8_mat_from_file("test_data", TEST_SZ, INPUT_SZ, path=path)

    train_label_org = load_uint8_vec_from_file("train_label", TRAIN_SZ, path=path)
    test_label = load_uint8_vec_from_file("test_label", TEST_SZ, path=path)

    ### normalise to -128/128
    training_data_org = np.asanyarray(np.asanyarray(training_data_org.tolist())-128, dtype=ctypes.c_int8)
    test_data = np.asanyarray(np.asanyarray(test_data.tolist())-128, dtype=ctypes.c_int8)

    ##### Prepare data augementation
    print("Data augmentation...")

    ### No data augmentation
    training_data = training_data_org
    train_label = train_label_org

    ### Elastic augmentation
    # aug_training_data = []
    # for i in range(0, TRAIN_SZ):
    #     aug_im = elastic_transform(np.reshape(training_data[i], (imRowL, imColL)), alpha=9, sigma=3)
    #     aug_training_data.append(np.reshape(aug_im, (INPUT_SZ)))
    #
    # aug_training_data = np.array(aug_training_data, dtype=ctypes.c_uint8)
    # training_data = np.concatenate((training_data, aug_training_data), axis=0)
    # TRAIN_SZ = len(training_data)
    # train_label = np.concatenate((train_label, train_label), axis=0)

    ##### Load files into data structures
    print("Preparing data structures...")

    for x in range(0, nAD):
        globals()[f"AD_fire_{x}"] = np.zeros(W, dtype=c_uint8)

    for x in SBC_COMB:
        globals()[f"D3_bmatrix{x[0]}{x[1]}"] = np.zeros(wwd, dtype=c_uint32)

    ##### Uncomment if KWTA FILTER
    for x in range(0, nAD):
        globals()[f"AD_row_sum{x}"] = np.zeros(W, dtype=c_int64)

    filter_pos_all, filter_coef_all, sizes_all = gen_rand_filter(nAD, INPUT_SZ, W)

    print("Save randomly generated filter data...")
    np.save(path + "/filter_pos_all_nAD{}_adW{}_{}.npy".format(nAD, W, "A"), filter_pos_all, allow_pickle=True)
    np.save(path + "/filter_coef_all_nAD{}_adW{}_{}.npy".format(nAD, W, "A"), filter_coef_all, allow_pickle=True)
    np.save(path + "/sizes_all{}_nAD_adW{}_{}.npy".format(nAD, W, "A"), sizes_all, allow_pickle=True)

    # filter_pos_all = np.load(path + "/filter_pos_all_nAD{}_adW{}_{}.npy".format(nAD, W, "B"), allow_pickle=True)
    # filter_coef_all = np.load(path + "/filter_coef_all_nAD{}_adW{}_{}.npy".format(nAD, W, "B"), allow_pickle=True)
    # sizes_all = np.load(path + "/sizes_all{}_nAD_adW{}_{}.npy".format(nAD, W, "B"), allow_pickle=True)

    for x in range(0, nAD):
        globals()[f"filter{x + 1}_pos"] = np.array(filter_pos_all[x], dtype=c_int16)
        globals()[f"filter{x + 1}_coef"] = np.array(filter_coef_all[x], dtype=c_int32)
        globals()[f"f{x + 1}sz"] = sizes_all[x]

    ##### Set up class variables
    class_count = np.zeros((SBC_CT, D), dtype=c_uint32)
    class_vector = np.zeros(D, dtype=c_uint32)
    confusion = np.zeros((D, D), dtype=c_uint32)
    wrong = 0

    ##### Prepare C code
    so_file = path + "Random_comp.so"
    lib = cdll.LoadLibrary(so_file)

    ##### Prepare find_AD_firing_pattern_KWTA_FILTER C code
    find_AD_firing_pattern_KWTA_FILTER_C = lib.find_AD_firing_pattern_KWTA_FILTER_C
    find_AD_firing_pattern_KWTA_FILTER_C.argtypes = [c_uint32, c_uint32, c_uint8, POINTER(c_int16), POINTER(c_int32), POINTER(POINTER(c_uint8)), POINTER(c_int64), POINTER(c_uint8)]

    UI8Ptr = POINTER(c_uint8)
    UI8PtrPtr = POINTER(UI8Ptr)
    ct_arr = np.ctypeslib.as_ctypes(training_data)
    UI8PtrArr = UI8Ptr * ct_arr._length_
    train_data_ptr = cast(UI8PtrArr(*(cast(row, UI8Ptr) for row in ct_arr)), UI8PtrPtr)

    ct_arr_test = np.ctypeslib.as_ctypes(test_data)
    UI8PtrArr = UI8Ptr * ct_arr_test._length_
    test_data_ptr = cast(UI8PtrArr(*(cast(row, UI8Ptr) for row in ct_arr_test)), UI8PtrPtr)

    ##### Prepare write_to_sbc C code
    write_to_sbc_C = lib.write_to_sbc_C
    write_to_sbc_C.argtypes = [POINTER(c_uint8), POINTER(c_uint8), POINTER(c_uint32),  c_uint32, c_uint8]

    ##### Prepare read_from_sbc C code
    read_from_sbc_C = lib.read_from_sbc_C
    read_from_sbc_C.argtypes = [POINTER(c_uint8), POINTER(c_uint8), POINTER(c_uint32),  c_uint32, POINTER(c_uint32)]

    ##### Supervised learning
    print("Supervised learning...")
    for i in range(0, TRAIN_SZ):
        if i % 1000 == 0:
            print("{}/{}".format(i, TRAIN_SZ))

        ##### write coincidences into SBC memories
        label = train_label[i]

        ##### find AD firing patterns
        for x in range(0, nAD):
            ### Random filter approach
            globals()[f"t{x}"] = threading.Thread(target=find_AD_firing_pattern_KWTA_FILTER_C,
                                                  args=(i, W, globals()[f"f{x+1}sz"], globals()[f"filter{x+1}_pos"].ctypes.data_as(POINTER(c_int16)),
                                                  globals()[f"filter{x+1}_coef"].ctypes.data_as(POINTER(c_int32)), train_data_ptr,
                                                  globals()[f"AD_row_sum{x}"].ctypes.data_as(POINTER(c_int64)), globals()[f"AD_fire_{x}"].ctypes.data_as(POINTER(c_uint8))))


            globals()[f"t{x}"].start()

        for x in range(0, nAD):
            globals()[f"t{x}"].join()

        t_idx = 0
        for x in SBC_COMB:
            globals()[f"t{t_idx}"] = threading.Thread(target=write_to_sbc_C, args=(globals()[f"AD_fire_{x[0]}"].ctypes.data_as(POINTER(c_uint8)), globals()[f"AD_fire_{x[1]}"].ctypes.data_as(POINTER(c_uint8)),
                                                                                   globals()[f"D3_bmatrix{x[0]}{x[1]}"].ctypes.data_as(POINTER(c_uint32)), W, label))
            globals()[f"t{t_idx}"].start()
            t_idx += 1

        t_idx = 0
        for x in SBC_COMB:
            globals()[f"t{t_idx}"].join()
            t_idx += 1

    ##### Save SBC
    # for x in SBC_COMB:
    #     np.save(path + "/D3_bmatrix{}{}_nAD{}_adW{}_{}.npy".format(x[0], x[1], nAD, W, "A"), globals()[f"D3_bmatrix{x[0]}{x[1]}"], allow_pickle=True)

    ##### Inference on test set
    print("Inference...")
    for i in range(0, TEST_SZ):
        if i % 1000 == 0:
            print("{}/{}".format(i, TEST_SZ))

        class_count = np.zeros((SBC_CT, D), dtype=c_uint32)

        for x in range(0, nAD):
            ### Random filter approach
            globals()[f"t{x}"] = threading.Thread(target=find_AD_firing_pattern_KWTA_FILTER_C,
                                                  args=(i, W, globals()[f"f{x+1}sz"], globals()[f"filter{x+1}_pos"].ctypes.data_as(POINTER(c_int16)),
                                                  globals()[f"filter{x+1}_coef"].ctypes.data_as(POINTER(c_int32)), test_data_ptr,
                                                  globals()[f"AD_row_sum{x}"].ctypes.data_as(POINTER(c_int64)), globals()[f"AD_fire_{x}"].ctypes.data_as(POINTER(c_uint8))))

            globals()[f"t{x}"].start()

        for x in range(0, nAD):
            globals()[f"t{x}"].join()

        ##### read coincidences from SBC memories
        t_idx = 0
        for x in SBC_COMB:
            globals()[f"t{t_idx}"] = threading.Thread(target=read_from_sbc_C, args=(globals()[f"AD_fire_{x[0]}"].ctypes.data_as(POINTER(c_uint8)), globals()[f"AD_fire_{x[1]}"].ctypes.data_as(POINTER(c_uint8)),
                                            globals()[f"D3_bmatrix{x[0]}{x[1]}"].ctypes.data_as(POINTER(c_uint32)), W, class_count[t_idx].ctypes.data_as(POINTER(c_uint32))))
            globals()[f"t{t_idx}"].start()
            t_idx += 1

        t_idx = 0
        for x in SBC_COMB:
            globals()[f"t{t_idx}"].join()
            t_idx += 1

        ##### collect counts
        class_vector = np.sum(class_count, axis=0)

        ##### find label of highest count
        inf_label = 0
        k = 0
        for j in range(0, D):
            if class_vector[j] > k:
                k = class_vector[j]
                inf_label = j

        ##### if not correct record it
        label = test_label[i]
        if inf_label != label:
            wrong += 1

        ##### build confusion matrix
        confusion[label][inf_label] += 1

    # print performance
    print("\n {} wrong = {}% correct \n".format(wrong, 100.0 - wrong / (TEST_SZ / 100.0)))

    print("Confusion matrix: ")
    print(confusion)

    print(" Finished in -- {} seconds, {} hours --".format((time.time() - start_time), (time.time() - start_time)/(60*60)))


def gen_rand_filter(nAD, INPUT_SZ, W):
    filter_pos_all = []
    filter_coef_all = []
    sizes = []

    for i in range(1, nAD+1):
        size = i * 500
        # size = i * 50
        # size = i * 10

        # filter_pos = np.random.randint(low=0, high=INPUT_SZ, size=size,  dtype=c_int16)
        filter_pos = np.random.randint(low=-INPUT_SZ, high=INPUT_SZ, size=size,  dtype=c_int16)

        filter_coef = np.random.randint(low=-50000, high=50000, size=size, dtype=c_int32)
        #filter_coef = np.random.randint(low=0, high=500, size=size, dtype=c_int32)

        sizes.append(size)
        filter_pos_all.append(filter_pos)
        filter_coef_all.append(filter_coef)

    return filter_pos_all, filter_coef_all, sizes


def elastic_transform(image, alpha, sigma, random_state=None, row=28, col=28):

    if len(image.shape) != 2:
        image = image.reshape(row, col)

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=1).flatten()


def rot_img(img):
    # dividing height and width by 2 to get the center of the image
    height, width = img.shape
    # get the center coordinates of the image to create the 2D rotation matrix
    center = (width / 2, height / 2)

    # make random angle
    ang = np.random.randint(1, 359)

    # using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=ang, scale=1)

    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))

    return rotated_image


def trans_img(img, tx=0, ty=0):
    height, width = img.shape

    # get tx and ty values for translation
    # you can specify any value of your choice
    # tx, ty = width / 4, height / 4

    # make random tx and ty
    if tx == 0:
        tx = np.random.randint(-1, 2)
    if ty == 0:
        ty = np.random.randint(-1, 2)

    # create the translation matrix using tx and ty, it is a NumPy array
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty]
    ], dtype=np.float32)

    translated_image = cv2.warpAffine(src=img, M=translation_matrix, dsize=(width, height))
    return translated_image

### Data loading functions
def load_uint8_mat_from_file(file_name, rows, cols, path="./", offset=0):
    return np.reshape(np.fromfile(path + file_name, dtype=ctypes.c_uint8)[offset:], (rows, cols))


def load_uint8_vec_from_file(file_name, size, path="./", offset=0):
    return np.fromfile(path + file_name, dtype=ctypes.c_uint8)[offset:]


def load_int32_mat_from_file(file_name, rows, cols, path="./"):
    return np.reshape(np.fromfile(path + file_name, dtype=ctypes.c_int32), (rows, cols))


def load_int32_vec_from_file(file_name, size, path="./"):
    return np.fromfile(path + file_name, dtype=ctypes.c_int32)

### MODIFIED
def load_int8_mat_from_file(file_name, rows, cols, path="./", offset=0):
    return np.reshape(np.fromfile(path + file_name, dtype=ctypes.c_int8)[offset:], (rows, cols))


def load_int8_vec_from_file(file_name, size, path="./", offset=0):
    return np.fromfile(path + file_name, dtype=ctypes.c_int8)[offset:]

if __name__ == '__main__':
    text_type = str
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--path',
        default='./',
        type=text_type,
        help='Default directory path',
    )

    args, unknown = parser.parse_known_args()
    try:
        main(args.path)
    except KeyboardInterrupt:
        pass
    finally:
        print()