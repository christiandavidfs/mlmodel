import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel
from typing import Optional

class MultiModalModel(nn.Module):
    def __init__(self, model_name: str = "gpt2", vision_model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.language_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.vision_model = AutoModel.from_pretrained(vision_model_name)
        
        self.vision_projection = nn.Linear(
            self.vision_model.config.hidden_size,
            self.language_model.config.hidden_size
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        
        # Get embeddings from text
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        
        # Get embeddings from images
        if pixel_values is not None:
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            image_features = vision_outputs.last_hidden_state
            image_embeddings = self.vision_projection(image_features)
            
            # Combine embeddings
            # For simplicity, we concatenate the embeddings
            # A more sophisticated approach would involve attention mechanisms
            combined_embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)
        else:
            combined_embeddings = text_embeddings

        # Adjust attention_mask if images are present
        if pixel_values is not None:
            image_attention_mask = torch.ones(image_embeddings.shape[:2], dtype=torch.long, device=input_ids.device)
            attention_mask = torch.cat([attention_mask, image_attention_mask], dim=1)

        return self.language_model(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
        )