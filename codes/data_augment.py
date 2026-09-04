import numpy as np
import pandas as pd
import json


import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, target_names=None, dataset=None, cmap=None, normalize=True, labels=True, title='Emotion Transition Matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(dataset+'_'+title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('label')
    plt.xlabel('label' ) 
    plt.show()

    
def file_read(dataset, mode):
    if dataset == 'MELD':
        f = open('/dataset/%s/multi/%s_%s.txt'%(dataset, dataset, mode),'r')
    else:
        f = open('/dataset/%s/%s_%s.txt'%(dataset, dataset, mode),'r')
    data = f.readlines()
    f.close

    return data

def transition_matrix_making_for_MELD(dataset):
    data_train = file_read(dataset,'train')

    dialogs = []
    temp_speakerList = []
    context = []
    context_emo = []
    context_speaker = []
    speakerNum = []
    emoSet = set()
    
    emodict = {'anger':0, 'disgust':1, 'fear':2, 'joy':3, 'neutral':4, 'sadness':5, 'surprise':6}
    emolist = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    for i, data in enumerate(data_train):
        if i <2:
            continue
        if data == '\n' and len(dialogs) > 0:
            speakerNum.append(len(temp_speakerList))
            temp_speakerList = []
            context= []
            context_emo = []
            context_speaker = []
            continue
        speaker, utt, emo, senti = data.strip().split('\t')
        context.append(utt)

        if speaker not in temp_speakerList:
            temp_speakerList.append(speaker)
        speakerCLS = temp_speakerList.index(speaker)
        context_speaker.append(speakerCLS)
        context_emo.append(emodict[emo])
        dialogs.append([context_speaker[:], context[:],context_emo[:], emodict[emo], senti])
        emoSet.add(emodict[emo])

    emo_list = []
    for i, dialog in enumerate(dialogs):
        if len(dialogs[i][0]) == 1 and i != 0:
            emo_list.append(dialogs[i-1][2])

    transition_matrix = np.zeros(shape=(len(emoSet),len(emoSet)))

    for idx, conversation in enumerate(emo_list):
        for i, data in enumerate(conversation):
            if i+1 != len(conversation):   
                transition_matrix[conversation[i]][conversation[i+1]] += 1

    transition_matrix = transition_matrix.astype('float') / transition_matrix.sum(axis=1)[:, np.newaxis]
    

    return transition_matrix

def transition_matrix_making(dataset):
    data_train = file_read(dataset,'train')

    dialogs = []
    temp_speakerList = []
    context = []
    context_emo = []
    context_speaker = []
    speakerNum = []
    emoSet = set()
    if dataset == 'iemocap':
        emodict = {'ang': 0, 'exc': 1, 'fru': 2, 'hap': 3, 'neu': 4, 'sad': 5}
        emolist = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
    elif dataset == 'dailydialog':
        emodict = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
        emolist = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    elif dataset == 'EMORY':
        emodict = {'Joyful':0, 'Mad':1, 'Neutral':2, 'Peaceful':3, 'Powerful':4, 'Sad':5, 'Scared':6}
        emolist = ['Joyful', 'Mad', 'Neutral', 'Peaceful', 'Powerful', 'Sad', 'Scared']
    
    for i, data in enumerate(data_train):
        if data == '\n' and len(dialogs) > 0:
            speakerNum.append(len(temp_speakerList))
            temp_speakerList = []
            context= []
            context_emo = []
            context_speaker = []
            continue
        speaker = data.strip().split('\t')[0]
        utt = ' '.join(data.strip().split('\t')[1:-1])
        emo = data.strip().split('\t')[-1]
        context.append(utt)

        if speaker not in temp_speakerList:
            temp_speakerList.append(speaker)
        speakerCLS = temp_speakerList.index(speaker)
        context_speaker.append(speakerCLS)
        context_emo.append(emodict[emo])
        dialogs.append([context_speaker[:], context[:],context_emo[:], emodict[emo]])
        emoSet.add(emodict[emo])

    emo_list = []
    for i, dialog in enumerate(dialogs):
        if len(dialogs[i][0]) == 1 and i != 0:
            emo_list.append(dialogs[i-1][2])

    transition_matrix = np.zeros(shape=(len(emoSet),len(emoSet)))

    for idx, conversation in enumerate(emo_list):
        for i, data in enumerate(conversation):
            if i+1 != len(conversation):   
                transition_matrix[conversation[i]][conversation[i+1]] += 1
    
    transition_matrix = transition_matrix.astype('float') / transition_matrix.sum(axis=1)[:, np.newaxis]
    

    return transition_matrix
