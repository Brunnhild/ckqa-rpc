# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline


class Pipeline(Text2TextGenerationPipeline):

    def _forward(self, model_inputs, **generate_kwargs):
        if self.framework == "pt":
            input_length = model_inputs["input_ids"].shape[-1]

        generate_kwargs["min_length"] = generate_kwargs.get("min_length", self.model.config.min_length)
        generate_kwargs["max_length"] = generate_kwargs.get("max_length", self.model.config.max_length)
        self.check_inputs(input_length, generate_kwargs["min_length"], generate_kwargs["max_length"])
        self.model.past_key_values = self.model.get_past_key_values()
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        return {"output_ids": output_ids}

    def postprocess(self, model_outputs, return_type=None, clean_up_tokenization_spaces=False):
        record = {
            "text": [self.tokenizer.decode(ids,
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                                           ) for ids in model_outputs["output_ids"]['sequences']],
            "score": [score.item() for score in model_outputs["output_ids"]['sequences_scores']]
        }
        return record


class Standalone:

    def __init__(self, model, tokenizer, max_step=10, device=-1):

        # update past_key_values
        model.past_key_values = model.get_past_key_values()
        model.eval()

        self.pipe = Pipeline(model=model, tokenizer=tokenizer, device=device)
        self.max_step = max_step

        self.end_token = 'NONE'

        self.template = "{context} </s> {history}"
        self.spo_sep_with_space = ' . '
        self.e_sep_with_space = ' ; '

    def get_batch_input(self, texts, extraction):
        batch_input = []
        for text in texts:
            exts = extraction[text]

            if len(exts) == 0:
                e = self.template.format(context=text, history=self.end_token)
            else:
                e = self.template.format(context=text, history=self.spo_sep_with_space.join(exts))

            batch_input.append(e)

        return batch_input

    @torch.no_grad()
    def pipeline(self, texts, batch_size):

        # quick pipeline
        extraction = {t: {} for t in texts}
        stack = texts[:]
        count = [0] * batch_size

        with tqdm(total=len(texts), desc='Inference ...', mininterval=60) as t:
            while stack:

                batch = stack[:batch_size]
                count += [0] * (len(batch) - len(count))
                count = [c + 1 for c in count]

                batch_input = self.get_batch_input(batch, extraction)
                outputs = self.pipe(batch_input,
                                    num_workers=0,
                                    # early_stopping=True,
                                    num_beams=5,  # Number of beams for beam search. 1 means no beam search.
                                    num_return_sequences=1,
                                    output_scores=True,  # Whether or not to return the prediction scores
                                    return_dict_in_generate=True,
                                    max_length=96,
                                    )

                # pop, if end of answer
                for i in range(len(batch) - 1, -1, -1):

                    text, output = batch[i], outputs[i]
                    answer_list, score_list = output['text'], output['score']

                    if any([answer == self.end_token for answer in answer_list]) or all(
                            [answer in extraction[text] for answer in answer_list]):

                        for answer, score in zip(answer_list, score_list):
                            if answer != self.end_token and answer not in extraction[text]:
                                extraction[text][answer] = score

                        del stack[i]
                        del count[i]
                        t.update()

                    else:
                        for answer, score in zip(answer_list, score_list):
                            if answer not in extraction[text]:
                                extraction[text][answer] = score

                        if count[i] == self.max_step:
                            del stack[i]
                            del count[i]
                            t.update()

        # decode
        extraction_decode = {t: {} for t in texts}
        for sent, extra in tqdm(extraction.items(), total=len(extraction), desc='Decoding Clause ...'):
            for clause, score in extra.items():

                seps = clause.split(self.e_sep_with_space.strip())
                if len(seps) != 5:
                    continue

                pred, arg0, arg1, place, time = seps
                pred = pred.strip()
                arg0 = arg0.strip()
                arg1 = arg1.strip()
                place = place.strip()
                time = time.strip()

                clause = (pred, arg0, arg1, place, time)
                if clause in extraction_decode[sent] and extraction_decode[sent][clause] < score:
                    continue

                extraction_decode[sent][clause] = score

        return extraction_decode
