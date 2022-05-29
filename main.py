# pose sequence as a NLI premise and label as a hypothesis
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

premise = '植物属于植物园。猫咪是一种可爱的种物。买植物MotivatedByGoal植物。猫猫是一种动物。植物是一种生物。植物是一种食物。稀有植物属于植物园。水生植物属于植物园。动物是一种物种。花是一种植物。茶是一种植物。草是一种植物。树是一种植物。狗是一种植物。植物与种相关。猫是一种生物。猫是一种动物。猫是一种宠物。'
hypothesis = '猫是一种植物'

# run through model pre-trained on MNLI
with torch.no_grad():
    x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                        truncation_strategy='only_first')
    logits = nli_model(x)[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true 
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:,1]

    print(prob_label_is_true.item())
