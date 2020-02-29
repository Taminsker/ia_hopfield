import hopfield as hop
import mod as mod
from pathlib import Path
import math
import os
import time

import variables as var # file of picture size and filename



def main(idx:str='0_flou', nb:int=30):
    """ idx: nom du fichier
        nb: nombre max d'itérations
    """

    myNetwork = hop.network()

    my_file = Path(var.WeightsMatrixFile)
    if my_file.is_file():
        print("The WeightsMatrix file exists. Please be patient we are loading it...")
        myNetwork.loadWeightsMatrix();
        print("     --> WeightsMatrix loaded")
    else:
        print("The WeightsMatrix file doesn't exist. Please be patient we are building it...\n")
        myNetwork.training_HebbRule(1);

        myNetwork.saveWeightsMatrix();

    my_file = Path(var.directoryPNG + "{}.png".format(idx))
    if my_file.is_file():
        pattern = mod.loadPattern(str(my_file), myNetwork.nmbrOfNeurons);
        myNetwork.iterationNetwork(pattern, nb);
    else:
        print('\n #####  the ' + str(my_file) + " file doesn't exist");


def osThing():
    choice = input('\nDo you want clean files ? (Matrix of Weights and all the models text files) : \n(y/n) \n >> ');

    if choice == 'y':
        if os.path.exists(var.directoryTXT):
            for file in os.listdir(var.directoryTXT):
                file_path = os.path.join(var.directoryTXT, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            os.rmdir(var.directoryTXT)
        if os.path.exists(var.WeightsMatrixFile):
            os.remove(var.WeightsMatrixFile)

    if not os.path.exists(var.directoryTXT):
        os.mkdir(var.directoryTXT)
        print("Directory " , var.directoryTXT ,  " created. ")


if __name__ == "__main__":
    os.system('clear');

    print('-----------------------------------------');
    print("     HOPFIELD'S NETWORK PROGRAM");
    print('-----------------------------------------\n');

    print('\nIn the variables file (variables.py) you have choose :\n');
    print('     <> Picture size on X axis : ' + str(var._picSize_X))
    print('     <> Picture size on Y axis : ' + str(var._picSize_Y))

    print('\nNumber of references files (' + str(var.lettersAvailable.size) + ')');
    osThing();
    main()

    print('\n-----------------------------------------');
    print('-----        END PROGRAM            -----');
    print('-----------------------------------------\n');
