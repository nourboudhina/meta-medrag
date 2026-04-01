import os, sys, torch  
sys.path.insert(0, r'C:\Users\Nour\Desktop\memoire\LLaVA-Med')  
from transformers import AutoTokenizer  
from llava.model import LlavaMistralForCausalLM  
tok = AutoTokenizer.from_pretrained('microsoft/llava-med-v1.5-mistral-7b', use_fast=False)  
model = LlavaMistralForCausalLM.from_pretrained('microsoft/llava-med-v1.5-mistral-7b', torch_dtype=torch.float32, device_map={'': 'cpu'})  
inputs = tok('What is pneumonia?', return_tensors='pt')  
with torch.no_grad(): out = model(**inputs, output_hidden_states=True)  
print(out.hidden_states[-2].shape)  
