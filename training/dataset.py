from PIL import Image
import glob
import argparse
import os
import random
import numpy as np
import pickle


### utils ###

# add / if there's not.
# @args
# pathName: str => path
#
# @return string
#
# example: 
# input: ./path/to/hoge
# output: ./path/to/hoge/
def addLastSlash(pathName):
    if not pathName.endswith("/"):
        return pathName + "/"

def save(ary, output):
    with open(output, "wb") as f:
        pickle.dump(ary, f)


### image processing ###

# Resize image k times as large as original.
# @args
# img: Image => image which will be resized.
# k: double => number of parameter which is how many times larger the input image will be.
#
# @return Image
def resize(img, k):
    w, h = img.size
    return img.resize(((int) (w * k), (int) (h * k)))

# get random position
# @args
# i: tuple => image's width and height
# r: tuple => rectangle's width and height.
#
# @return tuple of position
def randomRect(i, r):
    iW, iH = i
    rW, rH = r

    x = random.randint(0, max(iW - rW - 1, 0))
    y = random.randint(0, max(iH - rH - 1, 0))

    return (x, y)

# Crop 128 x 128 image from original image and resized image.
# @args
# img: Image => original image
# copy: Image => resized image
# count: how many cropped image this function will create.
#
# @return tuple of list of images
def randomCrop(img, copy, count):
    size = (128, 128)
    if size[0] > img.size[0] or size[1] > img.size[1]:
        print("can't crop an image...")
        return ([img], [copy])
    
    originals = []
    copies = []

    for _ in range(count):
        x, y = randomRect(img.size, size)

        originals.append(img.crop((x, y, x + size[0], y + size[1])))
        copies.append(copy.crop((x, y, x + size[0], y + size[1])))
    
    return (originals, copies)

# reform an image to numpy array
# @args
# img: Image
#
# return numpy array
def reform(img):
    nary = np.array(img, 'f')
    # Normalize the inputs -1.0 to 1.0
    nary[:,:] /= 128
    nary[:,:] -= 1.0

    return nary



# main
# @args
# args: Args => args.
#
# @return void.
def main(args):
    inputFolder = args.input
    outputFolder = "output"
    outputFile = "dataset.acdata"
    bulk = args.bulk

    #check inputFolder actually exists.
    if not os.path.exists(inputFolder):
        #if not, raise an error.
        print("input folder does not exist")
        exit()

    inputFolder = addLastSlash(inputFolder)
    
    #check outputFolder actually exists.
    if not os.path.exists("output"):
        #if not, make directories.
        os.makedirs("output")
    
    outputFolder = addLastSlash(outputFolder)

    result = []

    #suffix which this program is going to find.
    suffix = ["*.png", "*.gif", "*.jpg", "*.jpeg"]


    for i in range(len(suffix)):
        #get list of images
        files = glob.glob(inputFolder + suffix[i])

        for f in files:
            img = Image.open(f)
            copy = img.copy()
            copy = resize(copy, 1/4)
            copy = resize(copy, 4)

            originals, copies = randomCrop(img, copy, bulk)

            for j in range(len(originals)):
                ary = []
                x = copies[j]
                y = originals[j]

                ary = [reform(x), reform(y)]

                result.append(ary)
        
    
    save(result, outputFolder+outputFile)

    


if __name__ == "__main__":

    ### set args and get them ###
    parser = argparse.ArgumentParser(description="Create dataset")

    parser.add_argument("-i", "--input", type=str, default="dataset")
    parser.add_argument("--bulk", type=int, default=20)

    args = parser.parse_args()
    main(args)