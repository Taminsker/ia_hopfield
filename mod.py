from PIL import Image
import numpy as np
import math
import variables as var  # file of picture size and filename


def png2txtFile(_nmbrOfNeurons):

    for i in var.lettersAvailable:
        img = Image.open(var.directoryPNG + str(i));
        img = img.convert('L'); # L for luminescence
        caractere = np.fromiter(iter(img.getdata()), np.int) # convert pic to matrix
        caractere.resize(img.height, img.width); # resize to the good size the matrix

        caractere[caractere < 128] = 1;
        caractere[caractere >= 128] = -1;

        # caractere = -1 * caractere;

        arr = np.full((var._picSize_X, var._picSize_Y), -1, dtype=np.int); #dtype - data type

        arr = caractere;

        np.savetxt(var.directoryTXT + str(i) + '.txt', arr, delimiter=',', fmt='%i',newline ='\n'); # fmt = %i = format integer


def loadReferences(_nmbrOfNeurons):
    # lettersMaj = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']);
    # lettersMin = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']);
    # carac = np.array(['!', '#', '&', '*', '(', ')', '_', '-', '+',  '=', '[', ']', '\ ', '"', "'", ':', ';', '?', '.', '>', '<', ',']);
    # numbers = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']);
    # numbersreduce = np.array(['1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_', '9_', '0_']);


    # letters = np.concatenate((lettersMaj, lettersMin, carac, numbers, numbersreduce));
    # letters = np.array(['1', '2', '3', '4', '5', '6', '0']);
    # letters = lettersMaj;

    png2txtFile(_nmbrOfNeurons);

    arr = np.full((var.lettersAvailable.size, _nmbrOfNeurons), 0, dtype=np.float32); #
    for i in range(var.lettersAvailable.size):
        temp = np.loadtxt(var.directoryTXT + str(var.lettersAvailable[i]) + '.txt', delimiter = ',', dtype=np.float32);
        # print(temp[temp ==-1].size)
        arr[i,:] = temp.flatten(); # flatten = transform a matrix into a singular long array

    return arr;

def loadPattern(filename, _nmbrOfNeurons):

    img = Image.open(filename);
    img = img.convert('L');

    caractere = np.fromiter(iter(img.getdata()), np.float32)
    caractere.resize(img.height, img.width);


    caractere[caractere < 128] = 1;
    caractere[caractere >= 128] = -1;

    # caractere = -1 * caractere;

    arr = np.full((var._picSize_X, var._picSize_Y), -1, dtype=np.int)
    arr = caractere;

    temp = arr;
    temp = temp.astype(np.uint8)
    img = Image.fromarray(temp);
    img = img.resize((var._picSize_X * 7, var._picSize_Y * 7))
    img.show(title = "Picture after 0 iterations");
    # print(filename)
    # pause
    arr = arr.flatten();
    return arr;
