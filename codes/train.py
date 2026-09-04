import os
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from ERC_dataset import MELD_loader, Emory_loader, IEMOCAP_loader, DD_loader
from data_augment import transition_matrix_making_for_MELD, transition_matrix_making
from model import ERC_model

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import pdb
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support
from utils import encode_right_truncated, padding
from utils import make_batch_roberta, cluster
from supcon_loss import SupconLoss

import numpy as np, pandas as pd 
import zipfile

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt 
import seaborn as sns

import time

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def CELoss(pred_outs, labels):
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    loss_val = ce_loss(pred_outs, labels)
    return loss_val


def main():    
    batch_size = args.batch
    dataset = args.dataset
    dataclass = args.cls
    sample = args.sample
    model_type = args.pretrained
    loss_method = args.loss
    set_seed(args.seed)
    

    layer_set = args.layer_set
    print("layer_set: ", layer_set)
    
    if dataset == 'iemocap':
        emo_len = 6
        neu_label = 2
    elif dataset == 'EMORY':
        emo_len = 7
        neu_label = 2
    else:
        emo_len = 7
        neu_label = 4
    
    
    ConLoss = SupconLoss(emo_len, args.temp,neu_label, args)
    
    dataType = 'multi'
    if dataset == 'MELD':
        if args.dyadic:
            dataType = 'dyadic'
        else:
            dataType = 'multi'
        data_path = 'dataset/MELD/'+dataType+'/'
        DATA_loader = MELD_loader
    elif dataset == 'EMORY':
        data_path = 'dataset/EMORY/'
        DATA_loader = Emory_loader
    elif dataset == 'iemocap':
        data_path = 'dataset/iemocap/'
        DATA_loader = IEMOCAP_loader
    elif dataset == 'dailydialog':
        data_path = 'dataset/dailydialog/'
        DATA_loader = DD_loader
        
    if model_type == 'roberta-large':
        make_batch = make_batch_roberta
    

    train_path = data_path + dataset+'_train.txt'
    dev_path = data_path + dataset+'_dev.txt'
    test_path = data_path + dataset+'_test.txt'
    
    
    work_dir = ''
    
    train_dataset = DATA_loader(train_path, dataclass)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=make_batch)
    train_sample_num = int(len(train_dataset)*sample)
    
    dev_dataset = DATA_loader(dev_path, dataclass)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=make_batch)
    
    test_dataset = DATA_loader(test_path, dataclass)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=make_batch)
    
    if dataset == 'MELD':
        transition_matrix = transition_matrix_making_for_MELD(dataset)
    else:
        transition_matrix = transition_matrix_making(dataset)
    
    """logging and path"""
    save_path = os.path.join(dataset+'_models', model_type, args.loss)
    os.makedirs('test/diyi/temp', exist_ok=True)
    
         
    print("###Save Path### ", save_path)
    log_path = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)      
    
    last = False
        
    
    clsNum = len(train_dataset.labelList)
    model = ERC_model(model_type, clsNum, last, work_dir)
    model = model.cuda()
    model.train() 
    
    """Training Setting"""        
    training_epochs = args.epoch
    save_term = int(training_epochs/5)
    max_grad_norm = args.norm
    lr = args.lr
    
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    """Input & Label Setting"""
    best_dev_fscore, best_test_fscore = 0, 0
    best_dev_fscore_macro, best_dev_fscore_micro, best_test_fscore_macro, best_test_fscore_micro = 0, 0, 0, 0    
    best_epoch = 0
    

    patience = 0
    for epoch in tqdm(range(training_epochs),desc="Epoch: ", position=0, ncols=150 ):
        
        if args.augment == 'centroid':
            ''' Make Centroids '''
            centers, centers_mask, cluster2dataid, cluster2classid, all_centers = cluster(model, train_dataset, emo_len, args )
            centers = centers.cuda()
            
        for i_batch, data in (enumerate(tqdm(train_dataloader, desc="training: ", position=1, leave=False, ncols=150))):
            
            if i_batch > train_sample_num:
                print(i_batch, train_sample_num)
                break
            
            """Prediction"""
            if args.augment == 'centroid': 
                batch_input_tokens, batch_labels,_,_ = data
                batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()
            else: 
                batch_input_tokens, batch_labels,batch_aug_tokens, batch_aug_labels = data
                batch_input_tokens, batch_labels,batch_aug_tokens, batch_aug_labels = batch_input_tokens.cuda(), batch_labels.cuda(),batch_aug_tokens.cuda(), batch_aug_labels.cuda()

                model.eval()
                aug_pred_logits, aug_last_layer_embedding, aug_hidden_embeddings = model(batch_aug_tokens)
                
                
            model.train()  
            pred_logits, last_layer_embedding, hidden_embeddings = model(batch_input_tokens)
            
            if args.augment == 'centroid':
                '''Data Augmentation and Supervised Contrstive learning'''
                
                batch_regather = torch.zeros(args.batch, len(hidden_embeddings)-1, hidden_embeddings[0].size(-1)).cuda()
                
                for idx, h in enumerate(hidden_embeddings[1:]):
                    for id, hh in enumerate(h):
                        batch_regather[id][idx][:] = hh[0][:]
                
                augmentation = []
                aug_label = []
                
                for idx, batch_data in enumerate(zip(batch_regather,batch_labels)):
                    
                    hidden_embeddings, b_label = batch_data[0], batch_data[1]
                    
                
                    centers = centers.squeeze()

                    hidden_embedding_select = []
                    for layer in layer_set:
                        hidden_embedding_select.append(hidden_embeddings[layer,:] )

                    for h in hidden_embedding_select:
                        for i, t in enumerate(transition_matrix[b_label]):
                            if i != neu_label:
                                inter = (h*t) + (centers[i]*(1-t))
                                aug_label.append(i)
                                augmentation.append(inter)
                        

                aug_labels = torch.tensor(aug_label).cuda()
                aug_total = torch.zeros(1, 1024).cuda()

                for aug_rep in (augmentation):
                    aug_total = torch.cat((aug_total, aug_rep.unsqueeze(0)),0)
               
                aug_total = aug_total[1:]        
            
            
            """CE Loss calculation & training"""
            loss_ce = CELoss(pred_logits, batch_labels)
              
            if loss_method == 'ce':
                loss = loss_ce
            else:
                if args.augment != '':
                    loss_supcon = ConLoss(last_layer_embedding, batch_labels, aug_total, aug_labels)
                else:
                    loss_supcon = ConLoss(last_layer_embedding, batch_labels) 

                loss = loss_supcon + loss_ce
                
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        """Dev & Test evaluation"""
        model.eval()
        
        if dataset == 'dailydialog':
            dev_prek, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader,'dev')
            dev_pre_macro, dev_rec_macro, dev_fbeta_macro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='macro', zero_division = 0)
            dev_pre_micro, dev_rec_micro, dev_fbeta_micro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, labels=[0,1,2,3,5,6], average='micro', zero_division = 0)
            
            dev_fscore = dev_fbeta_macro+dev_fbeta_micro

            test_prek, test_pred_list, test_label_list = _CalACC(model, test_dataloader,'test', epoch, work_dir)
            test_pre_macro, test_rec_macro, test_fbeta_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro', zero_division = 0)
            test_pre_micro, test_rec_micro, test_fbeta_micro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, labels=[0,1,2,3,5,6], average='micro', zero_division = 0)
            """Best Score & Model Save"""
            if dev_fscore >= best_dev_fscore_macro + best_dev_fscore_micro:
                best_dev_fscore_macro = dev_fbeta_macro                
                best_dev_fscore_micro = dev_fbeta_micro
                best_test_fbeta_macro = test_fbeta_macro
                best_test_fbeta_micro = test_fbeta_micro
                
                best_epoch = epoch
                _SaveModel(model, save_path)     
            else:
                patience += 1      
              
        else:
            dev_prek, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader,'dev')
            dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted', zero_division = 0)
            test_prek, test_pred_list, test_label_list = _CalACC(model, test_dataloader, 'test',epoch, work_dir)
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted', zero_division = 0)                
            
            """Best Score & Model Save"""
            if dev_fbeta >= best_dev_fscore:
                best_dev_fscore = dev_fbeta
                best_test_fscore = test_fbeta
                best_test_prek = test_prek
                
                best_epoch = epoch
                _SaveModel(model, save_path)
            else:
                patience += 1
                
            
        logger.info('Epoch: {}'.format(epoch))
        if dataset == 'dailydialog':
            logger.info('Devleopment ## precision: {}, macro-fscore: {}, micro-fscore: {}'.format(dev_prek, dev_fbeta_macro, dev_fbeta_micro))
            logger.info('')
            print('Devleopment ## precision: {}, macro-fscore: {}, micro-fscore: {}'.format(dev_prek, dev_fbeta_macro, dev_fbeta_micro))
        else:
            logger.info('Devleopment ## precision: {}, precision: {}, recall: {}, fscore: {}'.format(dev_prek, dev_pre, dev_rec, dev_fbeta))
            logger.info('')
            print('Devleopment ## precision: {}, precision: {}, recall: {}, fscore: {}'.format(dev_prek, dev_pre, dev_rec, dev_fbeta))
            
        if patience > 4:
            print("Early stop!")
            break  
        
    if dataset == 'dailydialog':
        logger.info('Final Fscore ## test-precision: {}, test-macro: {}, test-micro: {}, test_epoch: {}'.format(best_test_prek, best_test_fbeta_macro, best_test_fbeta_micro, best_epoch)) 
    else:
        logger.info('Final Fscore ## test-precision: {}, test-fscore: {}, test_epoch: {}'.format(best_test_prek, best_test_fscore, best_epoch))            

def _CalACC(model, dataloader, mode, epoch =0, directory=''):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    tsne_embedding = []
    tsne_label = []
    p1num, p2num, p3num = 0, 0, 0    
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader, desc="Eval: ", position=1, leave=False, ncols=150)):            
            """Prediction"""
            batch_input_tokens, batch_labels,_,_ = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()
            
            pred_logits, last_layer_embedding,hidden_embedding = model(batch_input_tokens)
            
            """Calculation"""    
            pred_logits_sort = pred_logits.sort(descending=True)
            indices = pred_logits_sort.indices.tolist()[0]
            
            pred_label = indices[0]
            true_label = batch_labels.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
                
            """Calculation precision"""
            if true_label in indices[:1]:
                p1num += 1
            if true_label in indices[:2]:
                p2num += 1/2
            if true_label in indices[:3]:
                p3num += 1/3
            
        p1 = round(p1num/len(dataloader)*100, 2)
        p2 = round(p2num/len(dataloader)*100, 2)
        p3 = round(p3num/len(dataloader)*100, 2)
        
        
    return [p1, p2, p3], pred_list, label_list

def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )
    parser.add_argument( "--batch", type=int, help = "batch_size", default = 1)
    
    parser.add_argument( "--epoch", type=int, help = 'training epohcs', default = 10)
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-6)
    parser.add_argument( "--sample", type=float, help = "sampling trainign dataset", default = 1.0)
    parser.add_argument( "--dataset", help = 'MELD or EMORY or iemocap or dailydialog', default = 'MELD')
    parser.add_argument( "--pretrained", help = 'roberta-large or emoberta-large', default = 'roberta-large')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument( "--cls", help = 'emotion or sentiment', default = 'emotion')
    parser.add_argument( "--temp", help = 'temperature in contrastive learning loss', default = 0.05)    
    parser.add_argument( "--loss", help = 'method calculating loss. ce/neu/supcon  ',default = 'ce')
    parser.add_argument( "--tsne", help = 'want to visualization embedding space via T-SNE  ',default = False)
    parser.add_argument( "--prob", help = 'percentage of supcon+neu in loss', type=float,default = 0.5) 
    parser.add_argument( "--augment", help = 'data augmentation', choices=['centroid',''],type=str,default = 'centroid') 
    parser.add_argument( "--seed", type=int, help = "set seed", default = 2333)
    parser.add_argument("--pre_aug", type=bool, help="augmentation method by previous utterance of target", default = False)
    parser.add_argument( "--layer_set")    
    
    
    args = parser.parse_args()
    args.layer_set = [23]
    print(args)
    
    previous = args.pre_aug
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    
    main()
