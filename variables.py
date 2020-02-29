
import os
import numpy as np

_picSize_X = 15;
_picSize_Y = 27;

directoryPNG = "lettersPNG_" + str(_picSize_X) + 'x' + str(_picSize_Y) + '/';

directoryTXT = "lettersTXT_" + str(_picSize_X) + 'x' + str(_picSize_Y) + '/';

WeightsMatrixFile = 'weights_matrix_' + str(_picSize_X) + 'x' + str(_picSize_Y) + '.npz';


if not os.path.exists(directoryTXT):
    os.mkdir(directoryTXT);

###############################################################################

lettersMaj = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']);
lettersMin = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']);
carac =[]# np.array(['!', '#', '&', '*', '(', ')', '_', '-', '+',  '=', '[', ']', '\ ', '"', "'", ':', ';', '?', '.', '>', '<', ',']);
numbers = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']);
numbersreduce = np.array(['1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_', '9_', '0_']);


letters = np.concatenate((lettersMaj, lettersMin, carac, numbers, numbersreduce));
letters = [x + ".png" for x in letters]

lettersAvailable = np.array([], dtype = np.unicode_);

# print(letters)
# pause

for file in os.listdir(directoryPNG):
    # print(str(file))
    if file in letters:
        lettersAvailable = np.append(lettersAvailable, str(file));

lettersAvailable = lettersAvailable.flatten().astype(np.unicode_)
