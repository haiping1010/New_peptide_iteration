import copy
import numpy as np

import time


class SmilesTokenizer(object):
    def __init__(self):
        residues = list("ARNDCEQGHILKMFPSTWYV")
        special = [
            'X'
        ]
        padding = ['U', 'B', 'Z']

        self.table = sorted(residues, key=len, reverse=True)  + padding
        self.table_len = len(self.table)

        self.one_hot_dict = {}
        for i, symbol in enumerate(self.table):
            vec = np.zeros(self.table_len, dtype=np.float32)
            vec[i] = 1
            self.one_hot_dict[symbol] = vec

    def tokenize(self, smiles):
        N = len(smiles)
        i = 0
        token = []

        timeout = time.time() + 5   # 5 seconds from now
        while (i < N):
            for j in range(self.table_len):
                symbol = self.table[j]
                if symbol == smiles[i:i + len(symbol)]:
                    token.append(symbol)
                    i += len(symbol)
                    break
            if time.time() > timeout:
                break 
        return token

    def tokenize_matrix(self, smiles):

        N = len(smiles)
        nn=len(smiles[0])
        i = 0
        ii=0
        token = [0]*N
        for ix in range(N):
              token[ix]=[]
        timeout = time.time() + 5   # 5 seconds from now
        while (ii<N):
          i=0
          while (i < nn):
            for j in range(self.table_len):
                symbol = self.table[j]
                if symbol == smiles[ii][i:i + len(symbol)]:
                    token[ii].append(symbol)
                    #token[ii][i]=symbol
                    i += len(symbol)
                    break
            if time.time() > timeout:
                break

          ii=ii+1
        return token



    def one_hot_encode(self, tokenized_smiles):
        result = np.array(
            [self.one_hot_dict[symbol] for symbol in tokenized_smiles],
            dtype=np.float32)
        result = result.reshape(1, result.shape[0], result.shape[1])
        return result
    #added by zhanghaiping
    def one_hot_encode_matrix(self, tokenized_smiles):
        #print  (len(tokenized_smiles))
        result=np.zeros((len(tokenized_smiles), len(tokenized_smiles[0]), len(self.one_hot_dict['G'])), dtype=np.float32)
        for i, sentence in enumerate(tokenized_smiles):
           for t, char in enumerate(sentence):
                #XX[i, t, char_indices[char]] = 1
                result[i, t]=self.one_hot_dict[char]
        #print (tokenized_smiles)
        #result = np.array(
        #    [self.one_hot_dict[symbol]  for sequence in tokenized_smiles  for symbol in sequence] ,
        #    dtype=np.float32)

        #print (result.shape)
        #result = result.reshape(result.shape[0], result.shape[1],result.shape[2])
        return result

