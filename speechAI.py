import numpy as np
import theano as theano
import theano.tensor as T
import time
import operator
import shelve
import itertools
import os
import math
from copy import deepcopy


unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
post_start = "POST_START"
post_end = "POST_END"
input_v = 0
output_v = 1
forget_v = 2
cand_v = 3

class speechAI:
    def __init__(self, mem_Location):
        self.mem_Location = mem_Location
        self.load_Mem()

        self.theano = {}
        print("Compiling functions...")
        try:
            self.__theano_build__()
            print("Done\n")
        except:
            print("Error compiling functions\n")


    def load_Mem(self):
        print("Loading memories...")
        try:
            db = shelve.open("{0}/{0}DB".format(self.mem_Location), "r")
            self.E = db['E']
            self.U = db['U']
            self.W = db['W']
            self.b = db['b']
            self.V = db['V']
            self.d = db['d']
            self.w2i = db['w2i']
            self.i2w = db['i2w']
            self.hidden_dim = self.E.get_value().shape[0]
            db.close()
            print("Done\n")
        except:
            print("Memory loading failed\n")

            
    def _Step(self, z, C_Prev, s_t_Prev):
        #Just making things more legible
        E, U, W, b, V, d = self.E, self.U, self.W, self.b, self.V, self.d

        #Maps relationship between previous word and all known words.
        x_e = E[:,z]

        #LSTM Layer
        #Gate calculations; these modulate how much of the previous layer's data is used
        #The sigmoid function squishes the values to between 0 and 1
        f_t = T.nnet.hard_sigmoid(T.dot(U[forget_v], x_e) + T.dot(W[forget_v], s_t_Prev) + b[forget_v])
        i_t = T.nnet.hard_sigmoid(T.dot(U[input_v], x_e) + T.dot(W[input_v], s_t_Prev) + b[input_v])
        o_t = T.nnet.hard_sigmoid(T.dot(U[output_v], x_e) + T.dot(W[output_v], s_t_Prev) + b[output_v])

        #Candidate value for the cell memory that runs along the neural network
        C_c = T.tanh(T.dot(U[cand_v], x_e) + T.dot(W[cand_v], s_t_Prev) + b[cand_v])
        
        #Cell value actual
        C_t = i_t * C_c + f_t * C_Prev

        #New state value
        s_t = o_t * T.tanh(C_t)

        out = T.nnet.softmax(T.dot(s_t, V) + d)[0]

        return out, s_t, C_t

    def __theano_build__(self):

        x = T.ivector('x') #Input sequence stored as theano variable x
        

        print("Loading forward_step")                                          
        [out, s, C], updates = theano.scan(
            self._Step,
            sequences=x,
            truncate_gradient = 4,
            outputs_info=[None,
                          dict(initial=theano.shared(value = np.zeros(self.hidden_dim).astype(theano.config.floatX))),
                          dict(initial=theano.shared(value = np.ones(self.hidden_dim).astype(theano.config.floatX)))])

        
        pred = T.argmax(out, axis = 1)

        print("Loading f_pred")
        self.f_pred = theano.function([x], out) #Returns the predicted next word

    

    def make_Sent(self):
        test_Sent = [self.w2i[sentence_start_token]]
        wis = 0
        minLen = 5
        attempts = 0
        while test_Sent[-1] != self.w2i[sentence_end_token]:
            candidates = self.f_pred(test_Sent)[-1]
            samples = np.random.multinomial(1, candidates)
            new_Word = np.argmax(samples)
            wis += 1
            test_Sent.append(new_Word)
            if new_Word == self.w2i[sentence_end_token] and (len(test_Sent) - 2) < minLen:
                wis = 0
                test_Sent = [self.w2i[sentence_start_token]]
                attempts += 1
                if attempts > 50:
                    break
                continue
            if new_Word == self.w2i[unknown_token]:
                wis = 0
                test_Sent = [self.w2i[sentence_start_token]]
                continue
            if wis > 100:
                break

        sent = ''
        for i in test_Sent:
            if self.i2w[i] == sentence_start_token or self.i2w[i] == sentence_end_token:
                continue
            sent += self.i2w[i] + ' '
        if len(sent) > 0:
            return sent
        else:
            return "MAX ATTEMPTS REACHED"

if __name__ == "__main__":
    DB = input("DB: ")
    AI = speechAI(DB)
    while(True):
        try:
            print("Sample sentence:", AI.make_Sent())
            time.sleep(2)
        except:
            input("PAUSED. PRESS ENTER TO RESUME.")


        
