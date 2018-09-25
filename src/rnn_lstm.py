import os
import numpy as np
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout, Embedding
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_accuracy
from itertools import chain
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import layers
import matplotlib.pyplot as plt

'''
Trains a basic RNN and LSTM on the first five tasks of Facebook bABI.

Inspiration for this code is taken from the Keras team babi_rnn file.

Specifically: parse_stories and data_to_vector are taken from babi_rnn, credits
go to the Keras team

Original comes from "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
http://arxiv.org/abs/1502.05698

Task Number                  | FB LSTM Baseline | Keras QA
---                          | ---              | ---
QA1 - Single Supporting Fact | 50               | 100.0
QA2 - Two Supporting Facts   | 20               | 50.0
QA3 - Three Supporting Facts | 20               | 20.5
QA4 - Two Arg. Relations     | 61               | 62.9
QA5 - Three Arg. Relations   | 70               | 61.9
QA6 - yes/No Questions       | 48               | 50.7
QA7 - Counting               | 49               | 78.9
QA8 - Lists/Sets             | 45               | 77.2
QA9 - Simple Negation        | 64               | 64.0
QA10 - Indefinite Knowledge  | 44               | 47.7
QA11 - Basic Coreference     | 72               | 74.9
QA12 - Conjunction           | 74               | 76.4
QA13 - Compound Coreference  | 94               | 94.4
QA14 - Time Reasoning        | 27               | 34.8
QA15 - Basic Deduction       | 21               | 32.4
QA16 - Basic Induction       | 23               | 50.6
QA17 - Positional Reasoning  | 51               | 49.1
QA18 - Size Reasoning        | 52               | 90.8
QA19 - Path Finding          | 8                | 9.0
QA20 - Agent's Motivations   | 91               | 90.7

bAbI Project Resources:
https://research.facebook.com/researchers/1543934539189348:


'''
def setup_local_files():
    '''get files from local machine and return all training / testing text files in sorted order'''
    path = 'tasks'
    files = os.listdir(path)

    all_training_files = []
    all_testing_files = []

    for fn in files:
        if 'train' in fn:
            all_training_files.append(fn)
        if 'test' in fn:
            all_testing_files.append(fn)

    all_training_files = np.asarray(sorted(all_training_files))
    all_testing_files = np.asarray(sorted(all_testing_files))

    print(all_training_files)
    print(all_testing_files)

    return (all_training_files,all_testing_files)

# Setup local files
all_training_files,all_testing_files = setup_local_files()

def setup_dictionaries(training_files,testing_files):
    '''take in all training / testing files and return as dictionaries
    corresponding to tasks'''
    training_tasks_dict = dict((k+1,v) for k,v in enumerate(training_files))
    testing_tasks_dict = dict((k+1,v) for k,v in enumerate(testing_files))

    return (training_tasks_dict,testing_tasks_dict)

# Dictionary setup to grab tasks
training_tasks_dict,testing_tasks_dict = setup_dictionaries(all_training_files,all_testing_files)

def txt_to_raw(task_file):
    '''
    take in a specific task file and return a raw corpus
    '''
    with open(f'{os.getcwd()}/tasks/{task_file}', 'r') as file:
        raw_corpus = file.readlines()
        return raw_corpus

def parse_story(story):
    '''
    parse the passed in raw text corpus. This is modeled from the babi_rnn source from the Keras team.
    GitHub URL: https://github.com/keras-team/keras/blob/master/examples/babi_rnn.py
    '''
    related_content = []
    data = []

    for line in story:
        line_id,line = line.split(' ',1)
        line_id = int(line_id)

        if line_id == 1:
            related_content = []

        if '\t' in line:
            question,answer,supporting_facts = line.split('\t')
            question = text_to_word_sequence(question,filters='?\n')
            answer = [answer]
            substory = [ss for ss in related_content if ss]
            data.append((substory,question,answer))
            related_content.append('')
        else:
            line = text_to_word_sequence(line,filters='.\n') + ['.']

            for word in line:
                related_content.append(word)
    return data

def get_unique_vocab(train_file,test_file):
    '''opens up files and grabs unique vocabulary words from the text'''
    with open(f'{os.getcwd()}/tasks/{train_file}','r') as train_file, open(f'{os.getcwd()}/tasks/{test_file}','r') as test_file:
        raw_corpus_train = train_file.read()
        raw_corpus_test = test_file.read()

        train_tokenized = text_to_word_sequence(raw_corpus_train, filters='\n\t?123456789101112131415.')
        test_tokenized = text_to_word_sequence(raw_corpus_test, filters='\n\t?123456789101112131415.')
        return set(train_tokenized + test_tokenized + ['.'])

def data_to_vector(data,word_dictionary,vocab_size,sentence_limit,story_maxlen,question_maxlen):
    '''
    Stories and questions are represented as word embeddings and the answers are one-hot encoded.
    Takes the stories, finds unique words, and then vectorizing them into pure numeric form.
    Each word has a numeric index which it gets replaced by!

    This is modeled from the babi_rnn source from the Keras team.
    GitHub URL: https://github.com/keras-team/keras/blob/master/examples/babi_rnn.py
    '''
    STORY_VECTOR,QUESTION_VECTOR,ANSWER_VECTOR = [],[],[]

    for story,question,answer in data:
        # Encode the story representations
        STORY_VECTOR.append([word_dictionary[word] for word in story])
        # Encode the question representations
        QUESTION_VECTOR.append([word_dictionary[word] for word in question])
        ANSWER_VECTOR.append(word_dictionary[answer[0].lower()])

    return pad_sequences(STORY_VECTOR,maxlen=story_maxlen),pad_sequences(QUESTION_VECTOR,maxlen=question_maxlen),np.array(ANSWER_VECTOR)

def zip_sq(story_training_input,question_training_input,story_testing_input,question_testing_input):
    '''take story and question vectors and return a single
    concatenated vector for both training and testing alongside combined max length'''
    zipped_sq_training = list(zip(story_training_input,question_training_input))
    zipped_sq_testing = list(zip(story_testing_input,question_testing_input))

    sq_training_combined = []
    sq_testing_combined = []

    for sq in zipped_sq_training:
        sq_training_combined.append(list(chain(sq[0],sq[1])))

    for sq in zipped_sq_testing:
        sq_testing_combined.append(list(chain(sq[0],sq[1])))

    combined_maxlen = max(map(len,[sq for sq in sq_training_combined]))

    return (sq_training_combined,sq_testing_combined,combined_maxlen)

def build_rnn(combined_maxlen,vocab_maxlen,embedding_size,dropout_rate,learning_rate,task_num):
    '''build and return the model to be used'''

    print(f'Building, training and evaluating RNN for {task_num}\n\n')

    rnn_model = Sequential()
    rnn_model.add(Embedding(input_shape=combined_maxlen,input_dim=vocab_maxlen,output_dim=embedding_size))
    rnn_model.add(SimpleRNN(50,return_sequences=True))
    rnn_model.add(SimpleRNN(50))
    rnn_model.add(Dropout(dropout_rate))
    rnn_model.add(Dense(vocab_maxlen,activation='softmax'))

    rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

    print('Build completed, returning RNN Model...')

    return rnn_model

def run_rnn(rnn_model,x,y,testing_x,testing_y,epochs,task_num):
    '''build and run the rnn model and return the history'''

    print(f'Training and evaluating RNN for {task_num}\n\n')


    train_history = rnn_model.fit(x=np.array(x),y=np.array(y),
                                  epochs=epochs,validation_split=0.05)
    loss, accuracy = rnn_model.evaluate(x=np.array(testing_x),
                                        y=np.array(testing_y),
                                        batch_size=32)
    print(f'\n\nRNN Evaluation loss: {loss}, Evaluation accuracy: {accuracy} for task {task_num}\n\n')

    return train_history, loss, accuracy

def build_lstm(combined_maxlen,vocab_maxlen,embedding_size,dropout_rate,learning_rate,task_num):
    '''build and return the model to be used'''

    lstm_model = Sequential()
    lstm_model.add(Embedding(input_shape=combined_maxlen,input_dim=vocab_maxlen,output_dim=embedding_size))
    lstm_model.add(LSTM(50,return_sequences=True))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dropout(dropout_rate))
    lstm_model.add(Dense(vocab_maxlen, activation='softmax'))

    lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

    print('Build completed, returning LSTM Model...')

    return lstm_model

def run_lstm(lstm_model,x,y,testing_x,testing_y,epochs,task_num):
    '''build and run the lstm model'''

    print(f'Training and evaluating LSTM for {task_num}\n\n')


    train_history = lstm_model.fit(np.array(x),np.array(y),
                            epochs=epochs,validation_split=0.05)
    loss, accuracy = lstm_model.evaluate(x=np.array(testing_x),
                                        y=np.array(testing_y),
                                        batch_size=32)
    print(f'\n\nLSTM Evaluation loss: {loss}, Evaluation accuracy: {accuracy} for task {task_num}\n\n')

    return train_history, loss, accuracy

def predict_results(model,story_question_input,answer_testing_input):
    '''predict and return results of prediction'''
    def predictions_helper(expected,actuals):
        '''given the expected answers and the actual answers compare and contrast '''
        correct = 0

        for i in range(len(expected)):
            if expected[i] == actuals[i]:
                correct += 1

        print(f'\n\n----\nOut of 1000 possible answers the model correctly predicted: {correct}')

    predictions = model.predict([np.array(story_question_input)])

    idxs_of_preds = []

    for preds in predictions:
        for idx,ps in enumerate(preds):
            if ps == max(preds):
                idxs_of_preds.append(idx)

    print(f'List of all the predictions made by our Model: \n\n{idxs_of_preds}')
    print(f'\n\n---\n\n List of the expected values given by our testing: \n\n{answer_testing_input}')

    predictions_helper(answer_testing_input,idxs_of_preds)

def plot_loss(training_history, model_type, task_num):
    '''plot training vs validation loss'''
    plt.plot(training_history.history['loss'], label='Training Loss')
    plt.plot(training_history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type} Training loss vs Evaluation loss for task {task_num}')

def plot_acc(training_history, model_type, task_num):
    '''plot training vs validation accuracy'''
    plt.plot(training_history.history['acc'], label='Training Accuracy')
    plt.plot(training_history.history['val_acc'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_type} Training accuracy vs Evaluation accuracy for task {task_num}')

def plot_all_training_losses_rnn(rnn_hist):
    '''plot rnn training losses'''
    rnn_loss_epoch_fig = plt.figure().add_subplot(1,1,1)

    tasks = ['Single Supporting Fact', 'Two Supporting Facts', 'Three Supporting Facts',
            'Two Arg. Relations', 'Three Arg. Relations']

    for i in range(5):
        rnn_loss_epoch_fig.plot(rnn_hist[i].history['loss'], label=f'Task {i+1} - {tasks[i]}')

    rnn_loss_epoch_fig.legend()
    rnn_loss_epoch_fig.legend(bbox_to_anchor=(1, 1))
    rnn_loss_epoch_fig.set_xlabel('Epoch')
    rnn_loss_epoch_fig.set_ylabel('Loss')
    rnn_loss_epoch_fig.set_title(f'Loss rate for RNN for tasks 1 - 5 with Adam')

def plot_all_training_acc_rnn(rnn_hist):
    rnn_acc_fig = plt.figure().add_subplot(1,1,1)
    tasks = ['Single Supporting Fact', 'Two Supporting Facts', 'Three Supporting Facts',
            'Two Arg. Relations', 'Three Arg. Relations']

    for i in range(5):
        rnn_acc_fig.plot(rnn_hist[i].history['acc'], label=f'Task {i+1} - {tasks[i]}')

    rnn_acc_fig.legend(bbox_to_anchor=(1, 1))
    rnn_acc_fig.set_xlabel('Epoch')
    rnn_acc_fig.set_ylabel('Accuracy')
    rnn_acc_fig.set_title('Accuracy for RNN for tasks 1 - 5')

def plot_all_training_losses_lstm(lstm_hist):
    '''plot all lstm training losses'''
    lstm_loss_epoch_fig = plt.figure().add_subplot(1,1,1)
    tasks = ['Single Supporting Fact', 'Two Supporting Facts', 'Three Supporting Facts',
            'Two Arg. Relations', 'Three Arg. Relations']

    for i in range(5):
        lstm_loss_epoch_fig.plot(lstm_hist[i].history['loss'], label=f'Task {i+1} - {tasks[i]}')

    lstm_loss_epoch_fig.legend(bbox_to_anchor=(1, 1))
    lstm_loss_epoch_fig.set_xlabel('Epoch')
    lstm_loss_epoch_fig.set_ylabel('Loss')
    lstm_loss_epoch_fig.set_title('Loss rate for LSTM for tasks 1 - 5 with Adam')

def plot_all_training_acc_lstm(lstm_hist):
    lstm_acc_fig = plt.figure().add_subplot(1,1,1)
    tasks = ['Single Supporting Fact', 'Two Supporting Facts', 'Three Supporting Facts',
            'Two Arg. Relations', 'Three Arg. Relations']

    for i in range(5):
        lstm_acc_fig.plot(lstm_hist[i].history['acc'], label=f'Task {i+1} - {tasks[i]}')

    lstm_acc_fig.legend(bbox_to_anchor=(1, 1))
    lstm_acc_fig.set_xlabel('Epoch')
    lstm_acc_fig.set_ylabel('Accuracy')
    lstm_acc_fig.set_title('Accuracy for LSTM for tasks 1 - 5')

def run_all(embedding_size,dropout_rate,rnn_learning_rate,lstm_learning_rate,rnn_epochs,lstm_epochs):
    '''run all tasks and return history along with evaluations'''
    all_rnn_history = []
    all_lstm_history = []
    all_rnn_eval_loss = []
    all_lstm_eval_loss = []
    all_rnn_eval_acc = []
    all_lstm_eval_acc = []

    print('Running all tasks')
    print(f'Passed in parameters are the following EMBEDDING SIZE: {embedding_size}, DROPOUT RATE: {dropout_rate}',\
          f'LEARNING RATE FOR RNN: {rnn_learning_rate}, LEARNING RATE FOR LSTM: {lstm_learning_rate},\
          , RNN EPOCHS: {rnn_epochs}, LSTM EPOCHS: {lstm_epochs}\n\n')

    print('Building models...')

    for task_number in range(1,6):
        print(f'Running RNN and LSTM for Task {task_number}\n\n')
        # Text to raw
        task_training_corpus = txt_to_raw(training_tasks_dict[task_number])
        task_testing_corpus = txt_to_raw(training_tasks_dict[task_number])

        # Set up parsed stories
        training_data = parse_story(task_training_corpus)
        testing_data = parse_story(task_testing_corpus)

        # Get unique vocabulary
        vocab = get_unique_vocab(training_tasks_dict[task_number],testing_tasks_dict[task_number])


        # Get max lengths
        vocab_maxlen = len(vocab) + 1
        story_maxlen = max(map(len,[s for s,_,_ in training_data]))
        question_maxlen = max(map(len,[q for _,q,_ in training_data]))

        # Set up word indices
        word_index = dict((c, i + 1) for i, c in enumerate(vocab))

        index_words = [''] + list(vocab)

        # Vectorize stories, questions and answers
        vocab_maxlen = len(vocab) + 1
        sentence_limit = story_maxlen
        vocab_size = vocab_maxlen

        story_training_input,question_training_input,answer_training_input = data_to_vector(training_data,word_index,
                                                                                            vocab_size,sentence_limit,
                                                                                           story_maxlen,
                                                                                           question_maxlen)
        story_testing_input,question_testing_input,answer_testing_input = data_to_vector(testing_data,word_index,
                                                                                vocab_size,sentence_limit,
                                                                                        story_maxlen,
                                                                                        question_maxlen)

        # Zip up story, questions
        sq_training_combined,sq_testing_combined,combined_maxlen = zip_sq(story_training_input,question_training_input,
                                                 story_testing_input,question_testing_input)

        print('Building model, training and evaluating...\n\n')
        # Run and plot RNN / LSTM
        rnn_model = build_rnn(combined_maxlen=(combined_maxlen,),vocab_maxlen=vocab_maxlen,embedding_size=embedding_size,dropout_rate=dropout_rate,
                             learning_rate=rnn_learning_rate,task_num=task_number)
        lstm_model = build_lstm(combined_maxlen=(combined_maxlen,),vocab_maxlen=vocab_maxlen,embedding_size=embedding_size,dropout_rate=dropout_rate,
                               learning_rate=lstm_learning_rate,task_num=task_number)

        rnn_history, rnn_eval_loss, rnn_eval_acc = run_rnn(rnn_model=rnn_model,x=sq_training_combined,
                                                       y=answer_training_input,
                                                       testing_x=sq_testing_combined,
                                                       testing_y=answer_testing_input,
                                                       epochs=rnn_epochs,task_num=task_number)
        lstm_history, lstm_eval_loss, lstm_eval_acc = run_lstm(lstm_model=lstm_model,x=sq_training_combined,
                                                           y=answer_training_input,testing_x=sq_testing_combined,
                                                           testing_y=answer_testing_input,
                                                           epochs=lstm_epochs,task_num=task_number)

        # Make Predictions
        print(f'\n\n RNN Model Predictions for task {task_number}\n')
        rnn_predictions = predict_results(rnn_model, sq_testing_combined, answer_testing_input)
        print(f'\n\n LSTM Model Predictions for task {task_number}\n')
        lstm_predictions = predict_results(lstm_model, sq_testing_combined, answer_testing_input)

        all_rnn_history.append(rnn_history)
        all_lstm_history.append(lstm_history)
        all_rnn_eval_loss.append(rnn_eval_loss)
        all_rnn_eval_acc.append(rnn_eval_acc)
        all_lstm_eval_loss.append(lstm_eval_loss)
        all_lstm_eval_acc.append(lstm_eval_acc)

        print(f'End build for task {task_number}')

    return (all_rnn_history,all_lstm_history,
           all_rnn_eval_loss,all_rnn_eval_acc,
           all_lstm_eval_loss,all_lstm_eval_acc)


# All history for the model runs
all_history_evaluations = run_all(embedding_size=50,dropout_rate=0.10,rnn_learning_rate=0.0001,
                                 lstm_learning_rate=0.001,rnn_epochs=20,lstm_epochs=30)

# Separated histories for RNN / LSTM and Evaluation Loss / Accuracy
rnn_hist,lstm_hist,rnn_eval_loss,rnn_eval_acc,lstm_eval_loss,lstm_eval_acc = all_history_evaluations
