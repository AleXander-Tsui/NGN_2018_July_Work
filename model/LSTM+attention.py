from keras.models import *
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras import metrics
from pre_data import read_data
import numpy as np
import keras
class rate_callback(keras.callbacks.Callback):
    def __init__(self,training_data, validation_data):
        
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
    
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):        
        y_pred = np.argmax(self.model.predict(self.x),axis=2)
        j = 0
        for i in range(606):
            for m in range(len(y_pred[i])):
                if (self.y)[i][m] == 1:
                    mark = 0
                    for n in range(m,len(y_pred[i])):
                        if((self.y)[i][n]==0 or n==len(y_pred[i])):
                            mark=n
                            break
                    if ((self.y)[i][m:mark]==y_pred[i][m:mark]).all()==True:
                        j = j + 1    
        train_recall = j/1134
        n=0
        for i in range(606):
            for m in range(len(y_pred[i])):
                if y_pred[i][m]==1:
                    n = n + 1
        train_precision = j/n
        train_F1 = 2*train_precision*train_recall/(train_precision+train_recall)
        
        y_pred_val = np.argmax(self.model.predict(self.x_val),axis=2)
        j = 0
        for i in range(606):
            for m in range(len(y_pred_val[i])):
                if (self.y_val)[i][m] == 1:
                    mark = 0
                    for n in range(m,len(y_pred_val[i])):
                        if((self.y_val)[i][n]==0 or n==len(y_pred_val[i])):
                            mark=n
                            break
                    if ((self.y_val)[i][m:mark]==y_pred_val[i][m:mark]).all()==True:
                        j = j + 1
        test_recall = j/1134
        n=0
        for i in range(606):
            for m in range(len(y_pred_val[i])):
                if y_pred_val[i][m]==1:
                    n = n + 1
        test_precision = j/n
        test_F1 = 2*test_precision*test_recall/(test_precision+test_recall)
        print('\rtrain recall:',train_recall,' test recall:',test_recall,end=100*' '+'\n')
        print('\rtrain precison:',train_precision,' test precision:',test_precision,end=100*' '+'\n')
        print('\rtrain F1:',train_F1,' test F1:',test_F1,end=100*' '+'\n')
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return    

source_count, source_word2idx, target_count, target_word2idx = list(),{},list(),{}
train_data, train_loc,_,_,maxlen = read_data('./data/Restaurants_Train_v2.xml', source_count, source_word2idx, target_count, target_word2idx)
test_data, test_loc,_,_,_ = read_data('./data/Restaurants_Test_Gold.xml', source_count, source_word2idx, target_count, target_word2idx)

wt = np.random.normal(0, 0.05, [len(source_word2idx), 300])
with open('./data/glove.6B.300d.txt', 'r') as f:
    for line in f:
        content = line.strip().split()
        if content[0] in source_word2idx:
            wt[source_word2idx[content[0]]] = np.array([float(i) for i in content[1:]])

train_x = np.array(train_data)
train_y = np.array(train_loc)
test_x = np.array(test_data)
test_y = np.array(test_loc)
#label_y = to_categorical(test_y,num_classes=3)

train_x = pad_sequences(train_x,maxlen = 70)
train_y = pad_sequences(train_y,maxlen = 70)
test_x = pad_sequences(test_x,maxlen = 70)
test_y = pad_sequences(test_y,maxlen = 70)
label_train__y = to_categorical(train_y,num_classes=3)
label_test__y = to_categorical(test_y,num_classes=3)

def sum_axis_one(x):
    return K.sum(x,axis=1,keepdims=True)    
    
def attention_3d_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(70, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    output_sum = Lambda(sum_axis_one)(output_attention_mul)
    output_sum = Reshape((64,))(output_sum)
    output_repeat = RepeatVector(70)(output_sum)
    output = subtract([output_repeat,output_attention_mul])
    output = add([output,inputs])
    return output


inputs=Input(shape=(70,))
embedding=Embedding(5347, 300, dropout=0.2, input_length=70)(inputs)
lstm=LSTM(64,return_sequences=True,dropout=0.5)(embedding)
sentence=attention_3d_block(lstm)
#attention_add=Lambda(sum_except)(attention_mul)
#sentence=add([lstm,attention_add])
TD=TimeDistributed(Dense(3))(sentence)
output=Activation('softmax')(TD)
#Wx=Dense(64,activation='tanh')(lstm[:,69,:])
#type(lstm[:,69,:])
#h_star=Add()([Wp, Wx])
#output=Dense(3,activation='softmax')(Wp)
model=Model(inputs=inputs,outputs=output)
#model.summary()
#print(K.int_shape(lstm[:,69,:]))

model.layers[1].set_weights([wt])
model.layers[1].trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, label_train__y, batch_size=32, epochs=40,validation_data=(test_x,label_test__y),callbacks=[rate_callback(training_data=[train_x, train_y], validation_data=[test_x, test_y])])
