import torch
from torch import nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.gpt2.configuration_gpt2 import GPT2Config, logger
from transformers.modeling_utils import PreTrainedModel
from .modeling_llm import Block
from transformers import CLIPVisionModel, AutoProcessor, CLIPTextModel
from transformers.generation import GenerationMixin

class VisionProjector(nn.Module):  
    """Projette les caractéristiques visuelles dans l'espace du modèle de langage"""
    def __init__(self, input_dim=1024, output_dim=768): 
        super().__init__() 
        self.linear1 = nn.Linear(input_dim, output_dim*4) 
        self.gelu = nn.GELU() 
        self.linear2 = nn.Linear(output_dim*4, output_dim) 
 
    def forward(self, x): 
        return self.linear2(self.gelu(self.linear1(x))) 
# class VisionProjector(nn.Module):
#     def __init__(self, input_dim=1024, output_dim=768):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, output_dim * 4)
#         self.gelu = nn.GELU()
#         self.linear2 = nn.Linear(output_dim * 4, output_dim)
#         self.layer_norm = nn.LayerNorm(output_dim)  # Ajouté

#     def forward(self, x):
#         x = self.gelu(self.linear1(x))
#         x = self.linear2(x)
#         return self.layer_norm(x)  # Ajouté
class GPT2PreTrainedModel(PreTrainedModel):
    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

class VisionLanguageFusionLayer(GPT2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        # pour la fusion on va juste utiliser une couche et on reçoit directement les features textuelles extratites
        if config.num_hidden_layers != 1:
            self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Initialiser les poids
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs, # added
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        presents = () if use_cache else None
     
        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]
            
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)


        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents] if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
        )

class MultimodalFusionLayer(GPT2PreTrainedModel, GenerationMixin): 
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        
        # Initialiser les composants de base
        self.transformer = VisionLanguageFusionLayer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialiser les composants CLIP
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Initialiser le modèle CLIP
        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            low_cpu_mem_usage=False
        )

        # Initialiser le projecteur et la couche de fusion
        self.vision_projector = VisionProjector(
            input_dim=self.vision_model.config.hidden_size,
            output_dim=config.hidden_size
        )

        # Garder la configuration pour la couche de fusion pour pouvoir initialiser les poids
        fusion_layer_config = GPT2Config.from_dict(config.to_dict())
        fusion_layer_config.num_hidden_layers = 1
        self.fusion_layer = VisionLanguageFusionLayer(fusion_layer_config)

        # initialiser les poids
        self.post_init()

        # # Geler tout le modèle d'abord
        # self.requires_grad_(False)
        
        # # Dégeler uniquement les couches à entraîner
        # self.vision_projector.requires_grad_(True)
        # self.fusion_layer.requires_grad_(True)
        
        # self.transformer.requires_grad_(False)
        # self.vision_model.requires_grad_(False)
        # self.lm_head.requires_grad_(False)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    """
    redéfinition de la méthode prepare_inputs_for_generation pour prendre en compte les entrées multimodales
    key_values passées lors de la génération de séquences pour accélérer le processus
    """
    def prepare_inputs_for_generation( self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids  = kwargs.get("token_type_ids", None)
        # 
        if past_key_values: 
            past_length = past_key_values[0][0].shape[2] 
 
            # Quelques generation  méthodes passent déjà uniquement le dernier ID d'entrée
            if input_ids.shape[1] > past_length: 
                remove_prefix_length = past_length 
            else: 
                remove_prefix_length = input_ids.shape[1] - 1 
 
            input_ids = input_ids[:, remove_prefix_length:]  
            if token_type_ids is not None: 
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :] 
 
        attention_mask = kwargs.get("attention_mask", None) 
        position_ids = kwargs.get("position_ids", None) 
 
        if attention_mask is not None and position_ids is None: 
            position_ids = attention_mask.long().cumsum(-1) - 1 
            position_ids.masked_fill_(attention_mask == 0, 1) 
            if past_key_values: 
                position_ids = position_ids[:, -input_ids.shape[1] :] 
        else: 
            position_ids = None 
 
        if inputs_embeds is not None and past_key_values is None: 
            model_inputs = {"inputs_embeds": inputs_embeds} 
        else: 
            model_inputs = {"input_ids": input_ids} 
        model_inputs.update( 
            { 
                "past_key_values": past_key_values, 
                "use_cache": kwargs.get("use_cache"), 
                "position_ids": position_ids, 
                "attention_mask": attention_mask, 
                "token_type_ids": token_type_ids, 
            } 
        ) 
        return model_inputs 

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None, 
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        scores: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # added (carine)
        use_cache: Optional[bool] = None, # added
        token_type_ids: Optional[torch.LongTensor] = None, # added (carine)
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            return_dict=return_dict,
            position_ids=position_ids, # added
            use_cache=use_cache, # added 
            token_type_ids=token_type_ids, # added
        )
        hidden_states = transformer_outputs[0]
        logits_text = self.lm_head(hidden_states) 
 
        if pixel_values is not None: 
            # Prétraiter les images avec le processeur CLIP si les pixel_values ne sont pas déjà prétraités
            if not isinstance(pixel_values, torch.Tensor):
                pixel_values = self.processor(images=pixel_values, return_tensors="pt")["pixel_values"].to(hidden_states.device)
            
            if pixel_values.shape[0] != hidden_states.shape[0]:  
                hidden_states = hidden_states.repeat(pixel_values.shape[0], 1, 1) 
                input_ids = input_ids.repeat(pixel_values.shape[0], 1) 
                attention_mask = attention_mask.repeat(pixel_values.shape[0], 1) 
                if labels is not None: 
                    labels = labels.repeat(pixel_values.shape[0], 1) 
 
            if input_ids.shape[1] > 1: 
                self.fusion_key_values = None 
                vision_features = self.vision_model(pixel_values, output_hidden_states=True).hidden_states[-2][:, 1:, :] 
 
                vision_features = self.vision_projector(vision_features) 
                img_features_masks = torch.ones((vision_features.shape[0], vision_features.shape[1]), 
                                                device=attention_mask.device, dtype=attention_mask.dtype) 
                attention_mask = torch.cat([img_features_masks, attention_mask], dim=1) 
                inputs_embeds = torch.cat([vision_features, hidden_states], dim=1) 
                if labels is not None: 
                    labels_masks = torch.full((vision_features.shape[0], vision_features.shape[1]), fill_value=-100, 
                                              device=labels.device, dtype=labels.dtype) 
                    labels = torch.cat([labels_masks, labels], dim=1) 
 
                fusion_outputs = self.fusion_layer( 
                    inputs_embeds=inputs_embeds, 
                    attention_mask=attention_mask, 
                    return_dict=return_dict, 
                ) 
                hidden_states = fusion_outputs[0] 
                self.fusion_key_values = fusion_outputs.past_key_values 


        lm_logits = self.lm_head(hidden_states)

        if scores is not None: 
            print("************ SCORES CHECK **************")
            lm_logits = lm_logits[:, -logits_text.shape[1]:, :] 
            lm_logits = scores.view(-1, 1, 1) * lm_logits + (1- scores).view(-1, 1, 1) * logits_text.repeat(lm_logits.shape[0], 1, 1)
            lm_logits = lm_logits.mean(dim=0).unsqueeze(dim=0) 
            if labels is not None: 
                attention_mask = attention_mask[:1, -lm_logits.shape[1]:] 
                labels = labels[:1, -lm_logits.shape[1]:] 

        loss = None
        if labels is not None:
            # Decaler pour que les tokens <n prédisent n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:] 
                shift_logits = lm_logits[..., :-1, :][shift_attention_mask.to(lm_logits.device) != 0].contiguous() 
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous() 
            else: 
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

