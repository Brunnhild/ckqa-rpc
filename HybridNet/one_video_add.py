from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import numpy as np
import json
import random
import os
import sys

#加载模型roberta-large-nli-stsb-mean-tokens,前一个embedding用于query阶段，后一个用于找cms阶段
#torch.cuda.empty_cache()
def data_extract(one_caption):
    embedder = SentenceTransformer('/home/ubuntu/.cache/torch/sentence_transformers/sentence-transformers_roberta-large-nli-stsb-mean-tokens')
    embedder2 = SentenceTransformer('/home/ubuntu/.cache/torch/sentence_transformers/sentence-transformers_multi-qa-mpnet-base-dot-v1')
    
    #sys.path.append('./HybridNet/')
    #加载ATOMIC数据集，构建event组成的语料库
    #print("one_video_add:",os.getcwd())

    atomic_data=pd.read_csv("./HybridNet/data/atomic.csv")

    #获取ATOMIC中的各方面常识
    event=np.array(atomic_data["event"])
    intention=np.array(atomic_data["xIntent"])
    effect=np.array(atomic_data["xEffect"])
    attribute=np.array(atomic_data["xAttr"])
    need=np.array(atomic_data["xNeed"])
    react=np.array(atomic_data["xReact"])

    atomic_dict={}
    for i in range(len(event)):
        tmp={}
        eventi=event[i]
        intentioni=intention[i]
        effecti=effect[i]
        attributei=attribute[i]
        needi=need[i]
        reacti=react[i]
        tmp["intention"]=intentioni
        tmp["effect"]=effecti
        tmp["attribute"]=needi
        tmp["need"]=needi
        tmp["react"]=reacti
        atomic_dict[eventi]=tmp

    #将event赋值给语料库
    corpus=event

    #为语料库中的每个句子生成embedding
    #with torch.no_grad():
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    #原有的caption:
    queries = []
    captions=json.load(open('./HybridNet/data/caption_new1.json'))

    #单个caption待匹配

    #one_caption="a person is using a spoon to mix a dessert in a bowl"

    queries=[one_caption]

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(3, len(corpus))
    output=open('./HybridNet/query_res.txt',mode='w+')
    human_annote=0

    #按照数据集的形式保存最终的结果
    query_to_return=[]
    intention_final_data=[]
    effect_final_data=[]
    attribute_final_data=[]
    need_final_data=[]
    react_final_data=[]
    #cap_number=0
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        output.write("\n\n======================\n\n")
        print("\n\n======================\n\n")
        output.write("Query:")
        output.write(query)
        print("Query:", query)
        output.write("\nTop 3 most similar sentences in corpus:")
        print("\nTop 3 most similar sentences in corpus:\n")
        valid_num=0

        intention_Q=[]
        effect_Q=[]
        attribute_Q=[]
        need_Q=[]
        react_Q=[]

        intention_final=[]
        effect_final=[]
        attribute_final=[]
        need_final=[]
        react_final=[]
        for score, idx in zip(top_results[0], top_results[1]):
            if(score>=0.2):
                valid_num=valid_num+1
                output.write("\n")
                output.write(corpus[idx])
                query_to_return.append(corpus[idx])
                output.write("(Score: {:.4f})".format(score))
                output.write("\n")
                print(corpus[idx], "(Score: {:.4f})".format(score))

                intention_idx=atomic_dict[corpus[idx]]["intention"]
                effect_idx=atomic_dict[corpus[idx]]["effect"]
                attribute_idx=atomic_dict[corpus[idx]]["attribute"]
                need_idx=atomic_dict[corpus[idx]]["need"]
                react_idx=atomic_dict[corpus[idx]]["react"]

                #对得到的cms进行处理，每个方面分别进行处理

                #intention方面
                word=""
                word_cnt=0
                for i in range(len(intention_idx)):
                    if((intention_idx[i]>='A' and intention_idx[i]<='Z') or (intention_idx[i]>='a' and intention_idx[i]<='z') or (intention_idx[i]>='0' and intention_idx[i]<='9')):
                        word=word+intention_idx[i]
                    else:
                        if(word!=""):
                            if(intention_idx[i]!='\"'):
                                if(intention_idx[i]==' '):
                                    word=word+' '
                                else:
                                    continue
                            else:
                                word_cnt+=1
                                intention_Q.append(word)
                                word=""
                #effect方面
                word=""
                word_cnt=0
                for i in range(len(effect_idx)):
                    if((effect_idx[i]>='A' and effect_idx[i]<='Z') or (effect_idx[i]>='a' and effect_idx[i]<='z') or (effect_idx[i]>='0' and effect_idx[i]<='9')):
                        word=word+effect_idx[i]
                    else:
                        if(word!=""):
                            if(effect_idx[i]!='\"'):
                                if(effect_idx[i]==' '):
                                    word=word+' '
                                else:
                                    continue
                            else:
                                word_cnt+=1
                                effect_Q.append(word)
                                word=""
                #attribute方面
                word=""
                word_cnt=0
                for i in range(len(attribute_idx)):
                    if((attribute_idx[i]>='A' and attribute_idx[i]<='Z') or (attribute_idx[i]>='a' and attribute_idx[i]<='z') or (attribute_idx[i]>='0' and attribute_idx[i]<='9')):
                        word=word+attribute_idx[i]
                    else:
                        if(word!=""):
                            if(attribute_idx[i]!='\"'):
                                if(attribute_idx[i]==' '):
                                    word=word+' '
                                else:
                                    continue
                            else:
                                word_cnt+=1
                                attribute_Q.append(word)
                                word=""
                #need方面
                word=""
                word_cnt=0
                for i in range(len(need_idx)):
                    if((need_idx[i]>='A' and need_idx[i]<='Z') or (need_idx[i]>='a' and need_idx[i]<='z') or (need_idx[i]>='0' and need_idx[i]<='9')):
                        word=word+need_idx[i]
                    else:
                        if(word!=""):
                            if(need_idx[i]!='\"'):
                                if(need_idx[i]==' '):
                                    word=word+' '
                                else:
                                    continue
                            else:
                                word_cnt+=1
                                need_Q.append(word)
                                word=""
            #react方面
                word=""
                word_cnt=0
                for i in range(len(react_idx)):
                    if((react_idx[i]>='A' and react_idx[i]<='Z') or (react_idx[i]>='a' and react_idx[i]<='z') or (react_idx[i]>='0' and react_idx[i]<='9')):
                        word=word+react_idx[i]
                    else:
                        if(word!=""):
                            if(react_idx[i]!='\"'):
                                if(react_idx[i]==' '):
                                    word=word+' '
                                else:
                                    continue
                            else:
                                word_cnt+=1
                                react_Q.append(word)
                                word=""
        intention_Q=list(set(intention_Q))
        effect_Q=list(set(effect_Q))
        attribute_Q=list(set(attribute_Q))
        need_Q=list(set(need_Q))
        react_Q=list(set(react_Q))

        #各方面cms进行语义相似度匹配
        #intention方面
        query_embedding_intention = embedder2.encode(query, convert_to_tensor=True)
        corpus_intention=embedder2.encode(intention_Q, convert_to_tensor=True)
        cos_scores_intention = util.pytorch_cos_sim(query_embedding_intention, corpus_intention)[0]
        intention_k=min(3, len(intention_Q))
        top_results_intention = torch.topk(cos_scores_intention, k=intention_k)
        for score, idx in zip(top_results_intention[0], top_results_intention[1]):
            word=intention_Q[idx]
            word_list=word.split(' ')
            word_list.insert(0,'<sos>')
            word_list.append('<eos>')
            word_tmp=[]
            word_tmp.append(word)
            word_tmp.append(word_list)
            intention_final.append(word_tmp)
        #effect方面
        query_embedding_effect = embedder2.encode(query, convert_to_tensor=True)
        corpus_effect=embedder2.encode(effect_Q, convert_to_tensor=True)
        cos_scores_effect = util.pytorch_cos_sim(query_embedding_effect, corpus_effect)[0]
        effect_k=min(3, len(effect_Q))
        top_results_effect = torch.topk(cos_scores_effect, k=effect_k)
        for score, idx in zip(top_results_effect[0], top_results_effect[1]):
            word=effect_Q[idx]
            word_list=word.split(' ')
            word_list.insert(0,'<sos>')
            word_list.append('<eos>')
            word_tmp=[]
            word_tmp.append(word)
            word_tmp.append(word_list)
            effect_final.append(word_tmp)
        #attribute方面
        query_embedding_attribute = embedder2.encode(query, convert_to_tensor=True)
        corpus_attribute=embedder2.encode(attribute_Q, convert_to_tensor=True)
        cos_scores_attribute = util.pytorch_cos_sim(query_embedding_attribute, corpus_attribute)[0]
        attribute_k=min(3, len(attribute_Q))
        top_results_attribute = torch.topk(cos_scores_attribute, k=attribute_k)
        for score, idx in zip(top_results_attribute[0], top_results_attribute[1]):
            word=attribute_Q[idx]
            word_list=word.split(' ')
            word_list.insert(0,'<sos>')
            word_list.append('<eos>')
            word_tmp=[]
            word_tmp.append(word)
            word_tmp.append(word_list)
            attribute_final.append(word_tmp)
        #need方面
        query_embedding_need = embedder2.encode(query, convert_to_tensor=True)
        corpus_need=embedder2.encode(need_Q, convert_to_tensor=True)
        cos_scores_need = util.pytorch_cos_sim(query_embedding_need, corpus_need)[0]
        need_k=min(3, len(need_Q))
        top_results_need = torch.topk(cos_scores_need, k=need_k)
        #print("\n NEED TOP 3 \n")
        for score, idx in zip(top_results_need[0], top_results_need[1]):
            #print(need_Q[idx], "(Score: {:.4f})".format(score))
            word=need_Q[idx]
            word_list=word.split(' ')
            word_list.insert(0,'<sos>')
            word_list.append('<eos>')
            word_tmp=[]
            word_tmp.append(word)
            word_tmp.append(word_list)
            need_final.append(word_tmp)
        #react方面
        query_embedding_react = embedder2.encode(query, convert_to_tensor=True)
        corpus_react=embedder2.encode(react_Q, convert_to_tensor=True)
        cos_scores_react = util.pytorch_cos_sim(query_embedding_react, corpus_react)[0]
        react_k=min(3, len(react_Q))
        top_results_react = torch.topk(cos_scores_react, k=react_k)
        #print("\n NEED TOP 3 \n")
        for score, idx in zip(top_results_react[0], top_results_react[1]):
            #print(need_Q[idx], "(Score: {:.4f})".format(score))
            word=react_Q[idx]
            word_list=word.split(' ')
            word_list.insert(0,'<sos>')
            word_list.append('<eos>')
            word_tmp=[]
            word_tmp.append(word)
            word_tmp.append(word_list)
            react_final.append(word_tmp)

        intention_final_data.append(intention_final)
        effect_final_data.append(effect_final)
        attribute_final_data.append(attribute_final)
        need_final_data.append(need_final)
        react_final_data.append(react_final)
        #react_Q.append(react_idx)
        '''
                print("\nneed:",need_idx)
                print("\n")
                print("\nreact:",react_idx)
                print("\n")
        '''
        if(valid_num==0):
            human_annote=human_annote+1
            output.write("\n请人工标注\n")
            print("请人工标注")
        else:
            print("\n finally \n")
            #print("\nneed:",need_final)
            #print("\nreact:",react_final)
    output.write("\n人工标注数量：")
    output.write("(num: {:d})\n".format(human_annote))
    #print("\n finally \n")
    cap_num=0
    video_name="video30001"
    caption=[{}]
    caption[0]['caption']=one_caption
    word_list=one_caption.split(' ')
    word_list.insert(0,'<sos>')
    word_list.append('<eos>')
    caption[0]['final_caption']=word_list
    caption[0]['intention']=intention_final_data[0]
    caption[0]['effect']=effect_final_data[0]
    caption[0]['attribute']=attribute_final_data[0]
    caption[0]['need']=need_final_data[0]
    caption[0]['react']=react_final_data[0]
    captions[video_name]=caption
    '''for video in captions.keys():
        caption=captions[video]
        for iv in range(len(caption)):
            caption[iv]['need']=need_final_data[cap_num]
            caption[iv]['react']=react_final_data[cap_num]
        #print(caption)
        captions[video]=caption
        #print("\n---------------------------\n")
        cap_num=cap_num+1'''

    with open('./HybridNet/data/caption_new1.json', 'w') as f:
        json.dump(captions,f)
        f.close()
    print(query_to_return)
    print("-------------------------------------------step1 success--------------------------------------")
    return query_to_return
if __name__ == '__main__':
    data_extract("a person is using a spoon to mix a dessert in a bowl")
