# -*- coding: utf-8 -*-
from transformers import (AutoTokenizer,
                          AutoModelForQuestionAnswering, AutoModelForMaskedLM, AutoModelForCausalLM,
                          BertTokenizer, BartForConditionalGeneration,
                          top_k_top_p_filtering)
import torch


class SpanQA(torch.nn.Module):

    def __init__(self, pretrained_model_name_or_path):
        super(SpanQA, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path).eval()

    def forward(self, question, context):
        return self.predict(question, context)

    @torch.no_grad()
    def predict(self, question, context):
        inputs = self.tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        outputs = self.model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        # Get the most likely beginning of answer with the argmax of the score
        answer_start = torch.argmax(answer_start_scores)
        # Get the most likely end of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        return answer


class MaskedQA(torch.nn.Module):

    def __init__(self, pretrained_model_name_or_path, tokenizer=None):
        super(MaskedQA, self).__init__()
        if tokenizer:
            self.tokenizer = tokenizer.from_pretrained(pretrained_model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path).eval()

        self.mask_token = self.tokenizer.mask_token

    def forward(self, question, context):
        return self.predict(question, context)

    @torch.no_grad()
    def predict(self, question, context, top_k=5):
        input = self.tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
        mask_token_index = torch.where(input["input_ids"] == self.tokenizer.mask_token_id)[1]

        token_logits = self.model(**input).logits
        mask_token_logits = token_logits[0, mask_token_index, :]

        n = mask_token_logits.size(-1)
        top_5_indexes = torch.topk(mask_token_logits, min(top_k, n)).indices.tolist()
        top_5_tokens = [
            self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(top_5_indexes_)).strip().split(' ')
            for top_5_indexes_ in top_5_indexes]


        return [''.join(args) for args in zip(*top_5_tokens)]


class GuidedQA(torch.nn.Module):

    def __init__(self, pretrained_model_name_or_path):
        super(GuidedQA, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path).eval()

    def forward(self, question, context):
        return self.predict(question, context)

    @torch.no_grad()
    def predict(self, question, context):
        input_ids = self.tokenizer.encode(context + question, return_tensors="pt")

        # get logits of last hidden state
        next_token_logits = self.model(input_ids).logits[:, -1, :]

        # filter
        filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

        # sample
        probs = torch.softmax(filtered_next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        resulting_string = self.tokenizer.decode(next_token.tolist()[0])
        return resulting_string


class FreeQA(torch.nn.Module):

    def __init__(self, pretrained_model_name_or_path):
        super(FreeQA, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path).eval()

    def forward(self, question, context, top_k=10):
        return self.predict(question, context, top_k)

    @torch.no_grad()
    def predict(self, question, context, top_k):
        inputs = self.tokenizer(question, context, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs.input_ids,
                                      attention_mask=inputs.attention_mask,
                                      max_length=512,
                                      num_beams=top_k,
                                      num_return_sequences=top_k)
        ans = []
        start_mask = question.find('[MASK]')
        for output in outputs:
            tokens = self.tokenizer.convert_ids_to_tokens(output, skip_special_tokens=True)
            diff = len(tokens) - len(question) + 6
            ans.append(''.join(tokens[start_mask:start_mask + diff]))

        return ans
