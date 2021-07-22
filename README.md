<!-- omit in toc -->
# Commonsense knowledge for QA

- [1. References](#1-references)
- [2. Question Answering (QA) Setups](#2-question-answering-qa-setups)
- [3. *(GAP)* Context Generation](#3-gap-context-generation)
- [4. Context Retrieval Method](#4-context-retrieval-method)
- [5. Task Construction](#5-task-construction)
- [6. Details](#6-details)

## 1. References
- Language models as knowledge bases?. EMNLP. 2019. [[paper]](https://arxiv.org/abs/1909.01066)
- Realm: Retrieval-augmented language model pre-training. ICML. 2020. [[paper]](https://arxiv.org/abs/2002.08909)
- How context affects language models’ factual predictions. AKBC. 2020. [[paper]](https://www.akbc.ws/2020/papers/025X0zPfn)
- LAMA [[github]](https://github.com/facebookresearch/LAMA)

## 2. Question Answering (QA) Setups
1. **Masked Prediction(MP)**, we ask language models topredict single tokens in generic sentences *(RoBERTa-Large)*
2. **Free Generation(FG)**, we provide only questions, and letLMs generate arbitrary answer sentences *(GPT-2)*
3. **Guided Generation(GG)**, LMs are provided with an answersentence prefix. This provides a middle ground between the previous two setups, allowing multi-token answers, but avoiding some overly evasive answers *(GPT-2)*
4. **Span Prediction(SP)**, LMs select best answers from pro-vided content *(ALBERT-xxlarge on SQuAD 2.0)*

## 3. *(GAP)* Context Generation
From the [issue](https://github.com/phongnt570/ascent/issues/3)

To clarify, in the extrinsic evaluation, we only use pre-trained LMs without any fine-tuning. What we do is we only collect some relevant assertions from a CSKB and concatenate them with the given question, then we feed all of them to the LM (see also Table 4 in our paper for example, or you can play with our QA demo at https://ascent.mpi-inf.mpg.de/qa). All the LMs we used (i.e., RoBERTa, GPT-2 and ALBERT) could be found in the HuggingFace Transformers library (https://huggingface.co/models).

Yes, the LAMA project could be a good start for mask prediction evaluation. For the other QA settings (i.e., generation and span prediction), we only took the output of LMs and ask for human evaluation.

About converting assertions to natural language: We used an embarrassingly simple approach:

1. making subject plural: e.g., lion -> lions
2. convert "be" in predicates to "are"
3. concatenate subject, predicate, object
   
There are certainly better approaches for this. Obviously our approach will produce some grammar mistakes, but I would not worry much about it as the large pre-trained LMs can somehow deal with those mistakes decently.

For canonical schema like in ConceptNet, you can look at Table 9 in this paper (https://arxiv.org/pdf/2010.05953.pdf) for translation templates.

## 4. Context Retrieval Method
1. take into account assertions whose subjects are mentioned in the query
2. rank these assertions by the number of distinct tokens occurring in the input query
3. pick up the top ranked assertions and concate-nate them to build the contex

## 5. Task Construction
1. **Masked Prediction(MP)**, use the [CSLB](https://link.springer.com/article/10.3758%2Fs13428-013-0420-4) property norm dataset which consists of short human-written sentences about salient properties of general concepts. We hide the last token of each sentence, which is usually the object of that sentence. Besides, we remove sentences that contain less than three words.
   
2. **Free Generation(FG)** & **Guided Generation(GG)** & **Span Prediction(SP)**, 
use the Google Search Auto-completion functionality to collect commonsense questions about the aforementioned set of 150 engineering concepts, animals and occupations: For each subject, we feed the API with 6 prefixes: “what/when/where are/do \<subject\>”.

## 6. Details

**Requirements:**
> transformers == 4.6.0 \
> spacy == 3.1.0 \
> sentence_transformers == 1.1.1 \
> pytorch == 1.8.0 \
> scikit-learn == 0.24.1 \
> nltk == 3.5


1. 考虑中英文之间的差异，字级MASK在中文上可用性差，仅预测一个字且缺乏完整词语;
2. 实体识别基于词性分析，词典匹配对于大规模数据仍然是玩具级别；
3. 概念结点匹配停留在可用性高的字符串匹配；

