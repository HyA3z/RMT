import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, compress_module):
        super().__init__()
        self.model = base_model
        self.create_memory(num_mem_tokens)
        # add compress module
        self.compress_module = compress_module

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim =  getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, last_seg=False, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

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
        ''' if last_seg = True ----> use GPT-2 read memory tokens andgenerate result, 
            if last_seg = False ----> use compress module to compress information to memory tokens'''
        if inputs_embeds is None and last_seg: 
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)
        elif inputs_embeds is None and not last_seg: # Keep use shared wte. Can make sure tokens -----> same vector space
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
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

        seg_len = len(segmented) # segment_length
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
        last_logits = cell_outputs.logits # Only calculate last segment loss

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = last_logits[..., :-1, :].contiguous()
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

        out['logits'] = last_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        return out 
        
    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return memory_state
        
        memory_state = memory_state.detach()
        return memory_state