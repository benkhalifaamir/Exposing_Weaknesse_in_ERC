import torch
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, TensorDataset)
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
from augment_method import *

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

condition_token = ['<s1>', '<s2>', '<s3>'] 
special_tokens = {'additional_special_tokens': condition_token}
roberta_tokenizer.add_special_tokens(special_tokens)

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return [tokenizer.cls_token_id] + ids

def padding(ids_list, tokenizer, max_len=0):
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
        
    pad_ids = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        
        pad_ids.append(ids+add_ids)
    
    return torch.tensor(pad_ids), max_len

def encode_right_truncated_gpt(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return ids + [tokenizer.cls_token_id]

def padding_gpt(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        
        pad_ids.append(add_ids+ids)
    
    return torch.tensor(pad_ids)


def make_batch_roberta(sessions):
    batch_input, batch_labels = [], []
    batch_aug, batch_aug_labels = [], []
    
    for session in sessions:
        data = session[0]
        label_list = session[1]
        
        context_speaker, context, emotion, previous_emo = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []
        
        conversation_len = len(context_speaker)
        inputString = ""
        
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            
            inputString += '<s' + str(speaker+1) + '> '
            inputString += utt + " "
            
            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated(utt, roberta_tokenizer))


        Aug_inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            
            Aug_inputString += '<s' + str(speaker+1) + '> '
            Aug_inputString += utt + " "

        aug_concat_string = Aug_inputString.strip()
        concat_string = inputString.strip()
        
        batch_aug.append(encode_right_truncated(aug_concat_string, roberta_tokenizer))
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))
        
        label_ind = label_list.index(emotion)
        
        batch_labels.append(label_ind)        
        batch_aug_labels.append(label_ind)
        
    batch_input_tokens, max_len = padding(batch_input, roberta_tokenizer)
    batch_aug_tokens,_ = padding(batch_aug, roberta_tokenizer,max_len)
    
    batch_labels = torch.tensor(batch_labels)
    batch_aug_labels = torch.tensor(batch_aug_labels)
    
    return batch_input_tokens, batch_labels, batch_aug_tokens, batch_aug_labels


def gen_all_reps(model, data, args):
    
    model.eval()
    
    results = []
    label_results = []

    sampler = SequentialSampler(data)
    dataloader = DataLoader(
        data,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=4,
        collate_fn=make_batch_roberta
    )
    
    tq_train = tqdm(total=len(dataloader), position=1, ncols=50)
    tq_train.set_description("generate representations for all data")
    with torch.no_grad():
        for batch_id, batch_data in enumerate(dataloader):
            
            batch_data = [x.to(model.device()) for x in batch_data]
            
            sentences = batch_data[0]
            emotion_idxs = batch_data[1]
            outputs = model(sentences)
            _, outputs, _ = model(sentences)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            for idx, label in enumerate(emotion_idxs.reshape(-1)):
                if label < 0:
                    continue
                results.append(outputs[idx])
                label_results.append(label)
            tq_train.update()
    tq_train.close()
    dim = results[0].shape[-1]

    results = torch.stack(results, 0).reshape(-1, dim)
    label_results = torch.stack(label_results, 0).reshape(-1)

    return results, label_results

def cluster(model, data, emo_len, args):
    
    reps, labels = gen_all_reps(model, data, args)
    
    label_space = {}
    label_space_dataid = {}
    centers = []
    
    for idx in range(emo_len):
        label_space[idx] = []
        label_space_dataid[idx] = []
        
    for idx, turn_reps in enumerate(reps):
        emotion_label = labels[idx].item()
        if emotion_label < 0:
            continue
        label_space[emotion_label].append(turn_reps)
        label_space_dataid[emotion_label].append(idx)
        
    dim = label_space[0][0].shape[-1]

    max_num_clusters = 0
    cluster2dataid = {}
    cluster2classid = {}
    total_clusters = 0
    all_centers = []
    
    for emotion_label in range(emo_len):

        x = torch.stack(label_space[emotion_label], 0).reshape(-1, dim)
        
        num_clusters = 1
        cluster_idxs = torch.zeros(x.shape[0]).long()
        cluster_centers = x.mean(0).unsqueeze(0).cpu()
        
        centers.append(cluster_centers)

        max_num_clusters = max(num_clusters, max_num_clusters)
        
        cluster_idxs += total_clusters
        for d_idx, c_idx in enumerate(cluster_idxs.numpy().tolist()):
            if c_idx < 0:
                continue
            if cluster2dataid.get(c_idx) is None:
                cluster2dataid[c_idx] = []
            cluster2classid[c_idx] = emotion_label
            cluster2dataid[c_idx].append(
                label_space_dataid[emotion_label][d_idx])
        total_clusters += num_clusters
        for c_idx in range(num_clusters):
            all_centers.append(cluster_centers[c_idx, :])
    
    centers_mask = []
    for emotion_label in range(emo_len):
        num_clusters, dim = centers[emotion_label].shape[0], centers[
            emotion_label].shape[-1]
        centers_mask.append(torch.zeros(max_num_clusters))
        centers_mask[emotion_label][:num_clusters] = 1
        centers[emotion_label] = torch.cat(
            (centers[emotion_label],
             torch.ones(max_num_clusters - num_clusters, dim)), 0)
    centers = torch.stack(centers, 0).cuda()
    centers_mask = torch.stack(centers_mask, 0).cuda()
    print('Complete making centroid')
    return centers, centers_mask, cluster2dataid, cluster2classid, all_centers
