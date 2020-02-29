from PIL import Image
import numpy as np
from scipy import sparse
import scipy as scp
import time, sys
import mod as mod
import random as rand
import math

import variables as var  # file of picture size and filename

class network:
    """ Hopfield's network :
    - number of neurons
    - number of references
    - matrix of weights
    - matrix of references"""


    def __init__(self):
        """ Constructor """
        print("Create the Hopfield's Network\n");
        self.nmbrOfNeurons = var._picSize_X * var._picSize_Y;

        print('Number of Neurons : ' + str(self.nmbrOfNeurons));
        self.wSparseMatrix = sparse.lil_matrix((self.nmbrOfNeurons,self.nmbrOfNeurons)); # sparse matrix

        print("     Weights vector --> allocated        #float32 ");
        self.rMatrix = mod.loadReferences(self.nmbrOfNeurons);# ? x _nmbrOfNeurons
        self.nmbrOfReferences = self.rMatrix.size / self.nmbrOfNeurons;

        print("     References matrix --> allocated     #int\n");
        print("     Number of references : " + str(int(self.rMatrix.size / self.nmbrOfNeurons)) + '\n' );


    def training_HebbRule(self, nmbrOfTraining):
        """ Training program """
        for train in range(nmbrOfTraining):
            for i in range(self.nmbrOfNeurons):
                self.wSparseMatrix[i,(i+1):] += np.dot(self.rMatrix[:,i], self.rMatrix[:,(i+1):]) / float(self.nmbrOfReferences);

                sys.stdout.write("\r     Training " + str(train+1) + '/' + str(nmbrOfTraining) + ' presentations : ' + str(int((i+1) / self.nmbrOfNeurons * 100)) + "%     (Hebb rule)   ")

        print('\n     --> To change the matrix format will take few seconds.')
        self.wSparseMatrix = self.wSparseMatrix.tocsr()
        a = np.array(self.wSparseMatrix.todense());
        print('             matrix < 0 : '+ str(int(a[a < 0].size / a.size * 100)) + '%');
        print('             matrix = 0 : '+ str(int(a[a == 0].size / a.size * 100)) + '%');
        print('             matrix > 0 : '+ str(int(a[a > 0].size / a.size * 100)) + '%');

    def saveWeightsMatrix(self):
        print('     --> To save the matrix will take few seconds.')
        sparse.save_npz(var.WeightsMatrixFile, self.wSparseMatrix)
        self.wSparseMatrix += self.wSparseMatrix.T;


    def loadWeightsMatrix(self):
        self.wSparseMatrix = sparse.load_npz(var.WeightsMatrixFile)
        self.wSparseMatrix += self.wSparseMatrix.T;


    def update_neuron(self, pattern, neuronIndex):
        """ fait une mise à jour aléatoire des valeurs d'un pattern """
        _localPattern = pattern.copy()
        # la fonction de transfert
        fun = np.vectorize(lambda x: 1 if x >= 0 else -1)
        neuronNewValue = fun(np.dot(pattern, self.wSparseMatrix[:,neuronIndex].todense())) ##################

        # print('\nneuronIndex < 0.0 size : ' + str(neuronIndex[neuronIndex.any() < 0.0].size))
        # print('neuronIndex >= 0.0 size : ' + str(neuronIndex[neuronIndex.any() >= 0.0].size))

        #pattern[neuronIndex[neuronIndex.any() < 0.0]] = -1;
        #pattern[neuronIndex[neuronIndex.any() == 0.0]] = 1;
        #pattern[neuronIndex[neuronIndex.any() > 0.0]] = 1;

        # mise à jour de la copie locale sur les index calculés
        _localPattern[neuronIndex] = neuronNewValue
        return _localPattern

    def iterationNetwork(self, pattern, nmbrOfIterations):
        print("\nIteration of the network for the current picture || " + str(nmbrOfIterations) + " iter. max ||");

        for i in range(nmbrOfIterations):
            idRandNeuron = np.random.choice(self.nmbrOfNeurons, self.nmbrOfNeurons)
            ####### petit code de détection de point fixe ######
            _localPattern = self.update_neuron(pattern, idRandNeuron);

            sys.stdout.write("\r     Iterations " + str(int((i+1) / nmbrOfIterations * 100)) + "% ("+ str(i+1) + '/' + str(nmbrOfIterations) + ")");

            temp = _localPattern.reshape(var._picSize_Y, var._picSize_X)
            temp = temp.astype(np.uint8)
            img = Image.fromarray(np.array(temp));
            img = img.resize((var._picSize_X * 7, var._picSize_Y * 7))
            img.show(title = "Picture after " + str(i) + ' iterations');

            time.sleep(0.5);

            if np.all(_localPattern == pattern):
                sys.stdout.write("\n        Point fixe trouvé itération {}".format(i+1))
                break # on a trouvé stop
            else: pattern = _localPattern # on a pas trouvé .. update
            ######### fin modification mmc ######################
        print("     --> Iterations done");
