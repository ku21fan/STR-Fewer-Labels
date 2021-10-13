import os
import random

import fire
import lmdb
import cv2
import numpy as np
from tqdm import tqdm


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    # CAUTION: if outputPath (lmdb) already exists, this function add dataset
    # into it. so remove former one and re-create lmdb.
    if os.path.exists(outputPath):
        os.system(f"rm -r {outputPath}")

    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=30 * 2 ** 30)
    cache = {}
    cnt = 1

    with open(gtFile, "r", encoding="utf-8-sig") as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in tqdm(range(nSamples), total=nSamples, position=0, leave=True):
        imagePath, label = datalist[i].strip("\n").split("\t")
        imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print("%s does not exist" % imagePath)
            continue
        with open(imagePath, "rb") as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print("%s is not a valid image" % imagePath)
                    continue
            except:
                print("error occured", i)
                with open(outputPath + "/error_image_log.txt", "a") as log:
                    log.write("%s-th image data occured error\n" % str(i))
                continue

        imageKey = "image-%09d".encode() % cnt
        imagepathKey = "imagepath-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[imagepathKey] = imagePath.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            # print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache["num-samples".encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print("Created dataset with %d samples" % nSamples)


def createDataset_with_ValidTestset(
    inputPath,
    gtFile,
    outputPath,
    dataset_name,
    validset_percent=10,
    testset_percent=0,
    random_seed=1111,
    checkValid=True,
):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    train_path = os.path.join(outputPath, "training", dataset_name)
    valid_path = os.path.join(outputPath, "validation", dataset_name)

    # CAUTION: if train_path (lmdb) already exists, this function add dataset
    # into it. so remove former one and re-create lmdb.
    if os.path.exists(train_path):
        os.system(f"rm -r {train_path}")

    if os.path.exists(valid_path):
        os.system(f"rm -r {valid_path}")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)
    gt_train_path = gtFile.replace(".txt", "_train.txt")
    gt_valid_path = gtFile.replace(".txt", "_valid.txt")
    data_log = open(gt_train_path, "w", encoding="utf-8")

    if testset_percent != 0:
        test_path = os.path.join(outputPath, "evaluation", dataset_name)
        if os.path.exists(test_path):
            os.system(f"rm -r {test_path}")
        os.makedirs(test_path, exist_ok=True)
        gt_test_path = gtFile.replace(".txt", "_test.txt")

    env = lmdb.open(train_path, map_size=30 * 2 ** 30)
    cache = {}
    cnt = 1

    with open(gtFile, "r", encoding="utf-8-sig") as data:
        datalist = data.readlines()

    random.seed(random_seed)
    random.shuffle(datalist)

    nSamples = len(datalist)
    num_valid_dataset = int(nSamples * validset_percent / 100.0)
    num_test_dataset = int(nSamples * testset_percent / 100.0)
    num_train_dataset = nSamples - num_valid_dataset - num_test_dataset
    print(
        f"# Train dataset: {num_train_dataset}, # valid datast: {num_valid_dataset}, and # test datast: {num_test_dataset}"
    )

    for i in tqdm(range(nSamples), total=nSamples, position=0, leave=True):
        data_log.write(datalist[i])
        imagePath, label = datalist[i].strip("\n").split("\t")
        imagePath = os.path.join(inputPath, imagePath)

        if not os.path.exists(imagePath):
            print("%s does not exist" % imagePath)
            continue
        with open(imagePath, "rb") as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print("%s is not a valid image" % imagePath)
                    continue
            except:
                print("error occured", i)
                with open(outputPath + "/error_image_log.txt", "a") as log:
                    log.write("%s-th image data occured error\n" % str(i))
                continue

        imageKey = "image-%09d".encode() % cnt
        imagepathKey = "imagepath-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[imagepathKey] = imagePath.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            # print('Written %d / %d' % (cnt, nSamples))

        # Finish train dataset and Start validation dataset
        if i + 1 == num_train_dataset:
            print(f"# Train dataset: {num_train_dataset} is finished")
            cache["num-samples".encode()] = str(num_train_dataset).encode()
            writeCache(env, cache)
            data_log.close()

            # start validation set
            env = lmdb.open(valid_path, map_size=30 * 2 ** 30)
            cache = {}
            cnt = 0  # not 1 at this time
            data_log = open(gt_valid_path, "w", encoding="utf-8")

        # Finish train/valid dataset and Start test dataset
        if (i + 1 == num_train_dataset + num_valid_dataset) and num_test_dataset != 0:
            print(f"# Valid dataset: {num_valid_dataset} is finished")
            cache["num-samples".encode()] = str(num_valid_dataset).encode()
            writeCache(env, cache)
            data_log.close()

            # start test set
            env = lmdb.open(test_path, map_size=30 * 2 ** 30)
            cache = {}
            cnt = 0  # not 1 at this time
            data_log = open(gt_test_path, "w", encoding="utf-8")

        cnt += 1

    if testset_percent == 0:
        cache["num-samples".encode()] = str(num_valid_dataset).encode()
        writeCache(env, cache)
        print(f"# Valid datast: {num_valid_dataset} is finished")
    else:
        cache["num-samples".encode()] = str(num_test_dataset).encode()
        writeCache(env, cache)
        print(f"# Test datast: {num_test_dataset} is finished")


if __name__ == "__main__":
    fire.Fire(createDataset)
