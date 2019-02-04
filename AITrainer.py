import nltk
import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
from theano.tensor.opt import register_canonicalize
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
vocab_limit = 10000
input_v = 0
output_v = 1
forget_v = 2
cand_v = 3


class GradClip(theano.compile.ViewOp):

    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        
        return [T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound) for g_out in g_outs]

grad_clip = GradClip(-2., 2.)
register_canonicalize(theano.gof.OpRemove(grad_clip), name='grad_clip')

class RNNTheano:
    def __init__(self, word_dim, hidden_dim = 100, bptt_truncate = -1):
        self.num_Changed = 0 ###############
        self.vocab_limit = 3500
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.no_Change = []
        """
        The exact method of importing data will be determined later

        For now I'll just focus on the basic algorithm
        """
        
        self.w2i = {}

        self.test_name = input("Test name? ").strip()
        self.fileN = input("File name? ").strip()
        print("\nLoading voabulary...")
        self.build_Vocab(self.fileN)
        print("Done\n")
        try:
            db = shelve.open("{0}/{0}DB".format(self.test_name), "r")
            self.E = db['E']
            self.U = db['U']
            self.W = db['W']
            self.b = db['b']
            self.V = db['V']
            self.d = db['d']
            self.i2w = db['i2w']
            self.w2i = db['w2i']
            db.close()
        except:
            try:
                os.makedirs('{}'.format(self.test_name))
            except:
                pass
            db = shelve.open("{0}/{0}DB".format(self.test_name), "c")
            E = np.random.uniform(-np.sqrt(1./self.vocab_limit), np.sqrt(1./self.vocab_limit), (self.hidden_dim, self.vocab_limit))
            U = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (4, self.hidden_dim, self.hidden_dim))
            W = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (4, self.hidden_dim, self.hidden_dim))
            b = np.zeros((4, self.hidden_dim))
            V = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.vocab_limit))
            d = np.zeros(self.vocab_limit)
            db.close()
            
            #Theano Shared Variables
            self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
            self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
            self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
            self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
            self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
            self.d = theano.shared(name='d', value=d.astype(theano.config.floatX))
        finally:
            #Derivatives
            self.mE = theano.shared(name='mE', value=np.zeros(self.E.get_value().shape).astype(theano.config.floatX))
            self.mU = theano.shared(name='mU', value=np.zeros(self.U.get_value().shape).astype(theano.config.floatX))
            self.mW = theano.shared(name='mW', value=np.zeros(self.W.get_value().shape).astype(theano.config.floatX))
            self.mb = theano.shared(name='mb', value=np.zeros(self.b.get_value().shape).astype(theano.config.floatX))
            self.mV = theano.shared(name='mV', value=np.zeros(self.V.get_value().shape).astype(theano.config.floatX))
            self.md = theano.shared(name='md', value=np.zeros(self.d.get_value().shape).astype(theano.config.floatX))
            self.vE = theano.shared(name='vE', value=np.zeros(self.E.get_value().shape).astype(theano.config.floatX))
            self.vU = theano.shared(name='vU', value=np.zeros(self.U.get_value().shape).astype(theano.config.floatX))
            self.vW = theano.shared(name='vW', value=np.zeros(self.W.get_value().shape).astype(theano.config.floatX))
            self.vV = theano.shared(name='vV', value=np.zeros(self.V.get_value().shape).astype(theano.config.floatX))
            self.vb = theano.shared(name='vb', value=np.zeros(self.b.get_value().shape).astype(theano.config.floatX))
            self.vd = theano.shared(name='vd', value=np.zeros(self.d.get_value().shape).astype(theano.config.floatX))

        self.theano = {}
        print("Loading functions...")
        try:
            self.__theano_build__()
            print("Done\n")
            print("Building model...")
        except:
            print("Error loading functions")

    def build_Vocab(self, fileName):
        

        
        f = open('{}.txt'.format(fileName), 'r')
        reader = f.readlines()
        postFull = []
        for i in range(len(reader)):
            reader[i] = reader[i].strip()
            reader[i] = reader[i].replace('-', ' -')
            sentences = nltk.sent_tokenize(reader[i])
            postFull += sentences
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in postFull]
        print("Parsed %d sentences." % (len(sentences)))
             
        # Tokenize the sentences into words
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
         
        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))
        self.vocab_limit = min(self.vocab_limit, len(word_freq.items()))
        vocabulary_size = self.vocab_limit
         
        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(vocabulary_size-1)
        self.i2w = [x[0] for x in vocab]
        self.i2w.append(unknown_token)
        self.w2i = dict([(w,i) for i,w in enumerate(self.i2w)])
         
        print("Using vocabulary size %d." % vocabulary_size)
        print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
         
        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in self.w2i else unknown_token for w in sent]
         
        # Create the training data
        X_train = np.asarray([[self.w2i[w] for w in sent[:-1]] for sent in tokenized_sentences])
        Y_train = np.asarray([[self.w2i[w] for w in sent[1:]] for sent in tokenized_sentences])

        #Verification Data
        ver_perc_of_data = .95
        self.X_ver = X_train[len(X_train) * ver_perc_of_data:]
        self.X_train = X_train[0:len(X_train) * ver_perc_of_data]
        self.Y_ver = Y_train[len(Y_train) * ver_perc_of_data:]
        self.Y_train = Y_train[0:len(Y_train) * ver_perc_of_data]

        print("\nSample size: {}".format(len(self.X_train)))
        print("Verification size: {}".format(len(self.X_ver)))


    def forward_step(self, z, C_Prev, s_t_Prev):
        #Just making things more legible
        E, U, W, b, V, d = self.E, self.U, self.W, self.b, self.V, self.d

        #This maps the value passed to the function to the word's
        #unique weighted matrix. All known words have a weighted matrix
        #contained in E
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
        #Just making things more legible
        E, U, W, b, V, d = self.E, self.U, self.W, self.b, self.V, self.d

        x = T.ivector('x') #Input sequence stored as theano variable x
        y = T.ivector('y') #Target output value stored as theano variable y
        learnRate = T.scalar('learnRate')
        decayRate = T.scalar('decayRate')
        

        print("Loading forward_step")                                          
        [out, s, C], updates = theano.scan(
            self.forward_step,
            sequences=x,
            truncate_gradient = 4,
            outputs_info=[None,
                          dict(initial=theano.shared(value = np.zeros(self.hidden_dim).astype(theano.config.floatX))),
                          dict(initial=theano.shared(value = np.ones(self.hidden_dim).astype(theano.config.floatX)))])

        
        pred = T.argmax(out, axis = 1)

        #Predicts error of the output using categorical cross entropy
        pred_error = T.sum(T.nnet.categorical_crossentropy(out, y))

        print("Loading f_pred")
        self.f_pred = theano.function([x], out) #Returns the class
        self.f_pred_class = theano.function([x], pred)

        #Define function for calculating error
        print("Loading ce_error")
        self.ce_error = theano.function([x, y], pred_error, allow_input_downcast=True)

        print("Loading gradients")
        #Gradients
        dE = grad_clip(T.grad(pred_error, E))
        dW = grad_clip(T.grad(pred_error, W))
        dU = grad_clip(T.grad(pred_error, U))
        dV = grad_clip(T.grad(pred_error, V))
        db = grad_clip(T.grad(pred_error, b))
        dd = grad_clip(T.grad(pred_error, d))

        # Adam cache updates
        beta1 = .9
        beta2 = .999
        eps = 1e-8
        mE = grad_clip(beta1 * self.mE + (1 - beta1) * dE)
        mU = grad_clip(beta1 * self.mU + (1 - beta1) * dU)
        mW = grad_clip(beta1 * self.mW + (1 - beta1) * dW)
        mV = grad_clip(beta1 * self.mV + (1 - beta1) * dV)
        mb = grad_clip(beta1 * self.mb + (1 - beta1) * db)
        md = grad_clip(beta1 * self.md + (1 - beta1) * dd)
        vE = grad_clip(beta2 * self.vE + (1 - beta2) * (dE ** 2))
        vU = grad_clip(beta2 * self.vU + (1 - beta2) * (dU ** 2))
        vW = grad_clip(beta2 * self.vW + (1 - beta2) * (dW ** 2))
        vV = grad_clip(beta2 * self.vV + (1 - beta2) * (dV ** 2))
        vb = grad_clip(beta2 * self.vb + (1 - beta2) * (db ** 2))
        vd = grad_clip(beta2 * self.vd + (1 - beta2) * (dd ** 2))

        print("Loading adam_step")        
        self.adam_step = theano.function(
            [x, y, learnRate],
            [], 
            updates=[(E, E - learnRate * mE / (T.sqrt(vE) + eps)),
                     (U, U - learnRate * mU / (T.sqrt(vU) + eps)),
                     (W, W - learnRate * mW / (T.sqrt(vW) + eps)),
                     (V, V - learnRate * mV / (T.sqrt(vV) + eps)),
                     (b, b - learnRate * mb / (T.sqrt(vb) + eps)),
                     (d, d - learnRate * md / (T.sqrt(vd) + eps)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.md, md),
                     (self.vE, vE),
                     (self.vU, vU),
                     (self.vW, vW),
                     (self.vV, vV),
                     (self.vb, vb),
                     (self.vd, vd)
                    ], allow_input_downcast=True)

    def ver_Error(self):
        error = 0
        for sample in range(len(self.X_ver)):
            error += self.ce_error(self.X_ver[sample], self.Y_ver[sample])            
        return error/sample

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
                if attempts > 10:
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
            print("Sample sentence:", sent)
        else:
            print("Sample sentence: MAX ATTEMPTS REACHED")


def main():
    l = RNNTheano(8000)
    l_Rate = .001
    try:
        for epoch in range(1000):
            t_Start = time.time()
            for example in range(len(l.X_train)):
                l.adam_step(l.X_train[example], l.Y_train[example], l_Rate)
                if example % 1000 == 0:
                    print(example)
                    
##            print('DONE\n')
            t_End = time.time() - t_Start
            print("{}\nTime: {}".format(epoch, t_End))
            l.make_Sent()
            print()
    except KeyboardInterrupt:
        s = input("Save? ")
        if 'y' in s:
            db = shelve.open("{0}/{0}DB".format(l.test_name), "w")
            db['E'] = l.E
            db['U'] = l.U
            db['W'] = l.W
            db['b'] = l.b
            db['V'] = l.V
            db['d'] = l.d
            db['i2w'] = l.i2w
            db['w2i'] = l.w2i
            db.close()
    while True:
        l.make_Sent()
    
main()
