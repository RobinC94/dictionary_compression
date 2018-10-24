import sys, os
from termcolor import cprint
import keras
import argparse

from my_train_and_eval import evaluate_model, training_model
from my_models.modify_model import modify_model

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="model name", default="resnet50", type=str)
parser.add_argument("-t", "--train", help="whether train model", default=False, type=bool)
parser.add_argument("-s", "--save", help="whether save model", default=False, type=bool)
parser.add_argument("-e", "--epoches", help="train epoches", default=10, type=int)
parser.add_argument("-r", "--rate", help="comp rate", default=4, type=int)
parser.add_argument("-g", "--gpu", help="gpu id", default='', type=str)
args = parser.parse_args()



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    name = args.name
    if name == "resnet50":
        image_size = 224
        model = keras.applications.resnet50.ResNet50()
        model.load_weights("./weights/resnet50_weights_75_16.h5")
    elif name == "vgg16":
        image_size = 224
        model = keras.applications.vgg16.VGG16()
        model.load_weights("./weights/vgg16_weights_71_42.h5")
    elif name == "inceptionv3":
        image_size = 299
        model = keras.applications.inception_v3.InceptionV3()
    else:
        raise ValueError("model name wrong")

    model_new = modify_model(model, name=name, rate=args.rate, save=args.save)

    evaluate_model(model_new, name=name, image_size=image_size)
    #evaluate_model(model, name=name, image_size=image_size)

    if args.train:
        training_model(model, name=name, image_size=image_size, epoches=args.epoches, modified=True)