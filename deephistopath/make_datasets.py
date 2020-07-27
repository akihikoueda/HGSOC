import os, glob, math
import random
import shutil

## Input images information
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATION_DIR = os.path.join(BASE_DIR, "annotation2")
TRAIN_DIR = os.path.join(BASE_DIR,"labels","Train")
TEST_DIR = os.path.join(BASE_DIR,"labels","Test")
DATASET_NAME = "HGSOC"
IMAGE_CLASSES = ["IR", "MT", "SP", "PG", "ST"]
RATE_TRAIN = 0.85
RATE_TEST = 0.15

def num_train_test_tiles():
    num_tiles = []
    for idx,image_class in enumerate(IMAGE_CLASSES):
        image_dir = os.path.join(ANNOTATION_DIR,image_class)
        files = glob.glob(image_dir + '/*.jpg')
        num_tile = len(files)
        num_tiles.append(num_tile)
    train_tiles = math.floor(min(num_tiles)*0.85)
    test_tiles = math.floor(min(num_tiles)*0.15)
    return train_tiles, test_tiles

def train_test_split_dir(train_tiles, test_tiles):
    for idx,image_class in enumerate(IMAGE_CLASSES):
        image_dir = os.path.join(ANNOTATION_DIR,image_class)
        move_train_dir = TRAIN_DIR + "/" + image_class
        move_test_dir = TEST_DIR + "/" + image_class
        try:
            os.makedirs(move_train_dir, exist_ok=False)
            os.makedirs(move_test_dir, exist_ok=False)
            files = glob.glob(image_dir+'/*.jpg')
            random.shuffle(files)
            #move files to train dir
            for i in range(0, train_tiles):
                shutil.copy(files[i],move_train_dir)
            #move files to test dir
            for i in range(train_tiles, train_tiles+test_tiles):
                shutil.copy(files[i],move_test_dir)
        except FileExistsError:
            pass
        print('----{}を処理----'.format(image_class))

def main():
    # split train / test images
    train_tiles, test_tiles = num_train_test_tiles()
    print("Numbers of train and test tiles: train {}, test {}".format(train_tiles, test_tiles))
    train_test_split_dir(train_tiles, test_tiles)
    shutil.make_archive(os.path.join(BASE_DIR,'datasets',DATASET_NAME), 'zip', root_dir=BASE_DIR +"/labels/")
    print("Created datasets as {}".format(str(DATASET_NAME +".zip")))

if __name__ == "__main__":
    main()