import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch import nn
from transformers.activations import NewGELUActivation

class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, compress_module):
        super().__init__()
        self.model = base_model
        # add compress module
        self.compress_module = compress_module

        self.create_memory(num_mem_tokens)

        # Add Lora way to process memory
        self.A = nn.Linear(768, 256, bias=False, device=compress_module.device)
        self.B = nn.Linear(256, 768, bias=False, device=compress_module.device)
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.xavier_uniform_(self.B.weight)

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim =  getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

        # Its for last model to read
        self.read_memory_position = range(num_mem_tokens)
        # Its for compress module
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, last_seg=False, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)
            memory_state = self.A(memory_state)

        seg_kwargs = self.process_input(input_ids, memory_state, last_seg=last_seg, **kwargs)

        if last_seg:
            out = self.model(**seg_kwargs)
            out = self.process_output(out, last_seg=last_seg, **kwargs)
            return out
        else:
            out = self.compress_module(**seg_kwargs)
            new_memory_state = self.process_output(out, last_seg=last_seg, **kwargs)
            return new_memory_state
    

    def process_input(self, input_ids, memory_state, last_seg, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None and last_seg:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            # Transfer memory dim back 
            memory_state = self.B(memory_state)
            inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)
        elif inputs_embeds is None and not last_seg: # Compress process
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            # Transfer input_embeding dim to A (256)
            inputs_embeds = self.A(inputs_embeds)
            inputs_embeds = torch.cat([inputs_embeds, memory_state], dim=1)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape, last_seg=last_seg)
        seg_kwargs['output_hidden_states'] = True
        return seg_kwargs
    
    def pad_attention_mask(self, attention_mask, shape, last_seg):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            if last_seg:
                mask[:, self.num_mem_tokens:] = attention_mask
            else:
                mask[:, :-self.num_mem_tokens] = attention_mask                
            return mask
    
    def process_output(self, model_outputs, last_seg, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        if not last_seg:
            memory_state = model_outputs.last_hidden_state[:, -self.num_mem_tokens:]
            return memory_state 

        out['logits'] = model_outputs.logits[:, self.num_mem_tokens:]
        if kwargs.get('output_hidden_states'):
            out['hidden_states'] = [lh[:, self.num_mem_tokens:] for lh in model_outputs.hidden_states]
        if kwargs.get('output_attentions'):
            out['attentions'] = model_outputs['attentions']
        return out


import random
class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None, output_attentions=None, output_hidden_states=None):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # segment_length
        seg_len = len(segmented)
        for seg_num, segment in enumerate(segmented):
            last_seg = True if seg_num == seg_len - 1 else False

            if last_seg:
                cell_out = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True, last_seg=last_seg)
            else:
                memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True, last_seg=last_seg)
                memory_state = self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(cell_out, labels=labels,
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states)
        return out
    
    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments
    
    def split_tensor(self, tensor):
        align = self.rmt_config.get('segment_alignment')
        segment_size = self.rmt_config.get('segment_size')
        if align in {'left', None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align in {'right', None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align == 'center':
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = cell_outputs.logits
        # full_hidden_states = torch.cat([o.hidden_states for o in cell_outputs], dim=1)

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))

            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                
            out['loss'] = loss_fct(flat_logits, flat_labels)
        else:
            out['loss'] = 0

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        # if kwargs.get('output_hidden_states'):
        #     segment_keys.append('hidden_states')
        #     out['hidden_states'] = full_hidden_states

        # for seg_num, o in enumerate(cell_outputs):
        #     for key, value in o.items():
        #         if any([sk in key for sk in segment_keys]):
        #             out[f'{key}_{seg_num}'] = value
        return out 
        
    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return memory_state
        
        memory_state = memory_state.detach()
        return memory_state