# -*- coding: utf-8 -*-
from functools import partial

import torch.nn
from transformers import T5ForConditionalGeneration, T5Config


class PromptEmbedding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, projection=True, mid_dim=512, num_layers=6):
        super(PromptEmbedding, self).__init__()

        # num_layers * 2 (k & v) * embedding_dim
        if projection:
            self.embeds = torch.nn.Parameter(
                torch.FloatTensor(num_embeddings, embedding_dim).uniform_(-0.5, 0.5),
                requires_grad=True)
            self.stack = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, mid_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(mid_dim, num_layers * 2 * embedding_dim)
            )
        else:
            self.embeds = torch.nn.Parameter(
                torch.FloatTensor(num_embeddings, num_layers * 2 * embedding_dim).uniform_(-0.5, 0.5),
                requires_grad=True)

        self.projection = projection
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

    def forward(self, batch_size=1):
        embed = self.embeds

        if self.projection:
            embed = self.stack(embed)

        # embed = embed.repeat(batch_size, 1, 1)
        # embed = embed.view(batch_size, self.num_embeddings, self.num_layers * 2,
        #                    self.num_heads,
        #                    self.embedding_dim // self.num_heads)
        # embed = embed.permute([2, 0, 3, 1, 4]).split(2)

        embed = embed.unsqueeze(0)
        return embed.expand(batch_size, -1, -1) if batch_size != 1 else embed


class T5PromptTuningConfig(T5Config):

    def __init__(self, prompt_num_tokens=5, prompt_mid_dim=512, prompt_dropout=0.3, **kwargs):
        super(T5PromptTuningConfig, self).__init__(**kwargs)
        self.prompt_num_tokens = prompt_num_tokens
        self.prompt_mid_dim = prompt_mid_dim
        self.prompt_dropout = prompt_dropout


class T5PromptTuningForConditionalGeneration(T5ForConditionalGeneration):

    def __init__(self, config):
        super(T5PromptTuningForConditionalGeneration, self).__init__(config)

        for p in self.parameters():
            p.requires_grad = False

        self.encoder_embeds = PromptEmbedding(num_embeddings=config.prompt_num_tokens,
                                              embedding_dim=config.d_model,
                                              projection=True,
                                              mid_dim=config.prompt_mid_dim,
                                              num_layers=config.num_layers)
        self.encoder_dropout = torch.nn.Dropout(config.prompt_dropout)

        self.decoder_embeds = PromptEmbedding(num_embeddings=config.prompt_num_tokens,
                                              embedding_dim=config.d_model,
                                              projection=True,
                                              mid_dim=config.prompt_mid_dim,
                                              num_layers=config.num_decoder_layers)
        self.decoder_dropout = torch.nn.Dropout(config.prompt_dropout)

        self.modify_plm(self)

    def get_past_key_values(self, batch_size=1):
        encoder_past_key_values = self.encoder_embeds(batch_size)
        encoder_past_key_values = encoder_past_key_values.view(batch_size,
                                                               self.config.prompt_num_tokens,
                                                               self.config.num_layers * 2,
                                                               self.config.num_heads,
                                                               self.config.d_model // self.config.num_heads)
        encoder_past_key_values = self.encoder_dropout(encoder_past_key_values)
        encoder_past_key_values = encoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        decoder_past_key_values = self.decoder_embeds(batch_size)
        decoder_past_key_values = decoder_past_key_values.view(batch_size,
                                                               self.config.prompt_num_tokens,
                                                               self.config.num_decoder_layers * 2,
                                                               self.config.num_heads,
                                                               self.config.d_model // self.config.num_heads)
        decoder_past_key_values = self.decoder_dropout(decoder_past_key_values)
        decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return encoder_past_key_values, decoder_past_key_values

    def expand_to_batchsize(self, x, batch_size):
        return x.expand(-1, batch_size, -1, -1, -1)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):

        if self.training:
            # update prompt token cache every training step
            self.past_key_values = self.get_past_key_values()

        return super(T5PromptTuningForConditionalGeneration, self).forward(input_ids=input_ids,
                                                                           attention_mask=attention_mask,
                                                                           decoder_input_ids=decoder_input_ids,
                                                                           decoder_attention_mask=decoder_attention_mask,
                                                                           head_mask=head_mask,
                                                                           decoder_head_mask=decoder_head_mask,
                                                                           cross_attn_head_mask=cross_attn_head_mask,
                                                                           encoder_outputs=encoder_outputs,
                                                                           past_key_values=past_key_values,
                                                                           inputs_embeds=inputs_embeds,
                                                                           decoder_inputs_embeds=decoder_inputs_embeds,
                                                                           labels=labels,
                                                                           use_cache=use_cache,
                                                                           output_attentions=output_attentions,
                                                                           output_hidden_states=output_hidden_states,
                                                                           return_dict=return_dict)

    def modify_plm(self, model):
        # modify function 'forward' of SelfAttentionLayer
        backup_encoder_forward_functions = []
        for i, layer_module in enumerate(model.encoder.block):
            backup_encoder_forward_functions.append(layer_module.layer[0].forward)

            def modified_encoder_forward(*args, **kwargs):
                layer_id = kwargs.pop('layer_id')
                batch_size = args[0].shape[0]
                past_key_values = self.past_key_values[0][layer_id]

                if kwargs['past_key_value'] is None:
                    kwargs['past_key_value'] = self.expand_to_batchsize(past_key_values, batch_size)

                if kwargs['attention_mask'] is not None:
                    am = kwargs[
                        'attention_mask']  # Should check the format of the attention_mask when moving to a new plm.
                    kwargs['attention_mask'] = torch.cat(
                        [-torch.zeros((*am.shape[:-1], self.config.prompt_num_tokens), dtype=am.dtype,
                                      device=am.device), am],
                        dim=-1)

                return backup_encoder_forward_functions[layer_id](*args, **kwargs)

            layer_module.layer[0].forward = partial(modified_encoder_forward, layer_id=i)

        backup_decoder_forward_functions = []
        for i, layer_module in enumerate(model.decoder.block):
            backup_decoder_forward_functions.append(layer_module.layer[0].forward)

            def modified_decoder_forward(*args, **kwargs):
                batch_size = args[0].shape[0]
                layer_id = kwargs.pop('layer_id')
                past_key_values = self.past_key_values[1][layer_id]

                if kwargs['past_key_value'] is None:
                    kwargs['past_key_value'] = self.expand_to_batchsize(past_key_values, batch_size)

                if kwargs['attention_mask'].size(-1) == kwargs['past_key_value'][0].size(-2) + args[0].size(-2):
                    pass

                elif kwargs['attention_mask'].size(-1) + self.config.prompt_num_tokens == \
                        kwargs['past_key_value'][0].size(-2) + args[0].size(-2):

                    am = kwargs['attention_mask']
                    kwargs['attention_mask'] = torch.cat(
                        [torch.zeros((*am.shape[:-1], self.config.prompt_num_tokens), dtype=am.dtype, device=am.device),
                         am],
                        dim=-1)

                else:
                    raise RuntimeError("Size not match: past length: {}, input length:{},\
                               attention mask length {}".format(kwargs['past_key_value'][0].size(-2),
                                                                args[0].size(-2),
                                                                kwargs['attention_mask'].size(-1)))

                return backup_decoder_forward_functions[layer_id](*args, **kwargs)

            layer_module.layer[0].forward = partial(modified_decoder_forward, layer_id=i)

    @torch.no_grad()
    def generate(self,
                 input_ids=None,
                 max_length=None,
                 min_length=None,
                 do_sample=None,
                 early_stopping=None,
                 num_beams=None,
                 temperature=None,
                 top_k=None,
                 top_p=None,
                 repetition_penalty=None,
                 bad_words_ids=None,
                 bos_token_id=None,
                 pad_token_id=None,
                 eos_token_id=None,
                 length_penalty=None,
                 no_repeat_ngram_size=None,
                 encoder_no_repeat_ngram_size=None,
                 num_return_sequences=None,
                 max_time=None,
                 max_new_tokens=None,
                 decoder_start_token_id=None,
                 use_cache=None,
                 num_beam_groups=None,
                 diversity_penalty=None,
                 prefix_allowed_tokens_fn=None,
                 output_attentions=None,
                 output_hidden_states=None,
                 output_scores=None,
                 return_dict_in_generate=None,
                 forced_bos_token_id=None,
                 forced_eos_token_id=None,
                 remove_invalid_values=None,
                 synced_gpus=None,
                 **model_kwargs):

        # update prompt token cache before generation
        if not hasattr(self, 'past_key_values'):
            self.past_key_values = self.get_past_key_values()

        return super(T5PromptTuningForConditionalGeneration, self).generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            max_time=max_time,
            max_new_tokens=max_new_tokens,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=use_cache,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            remove_invalid_values=remove_invalid_values,
            synced_gpus=synced_gpus,
            **model_kwargs
        )
