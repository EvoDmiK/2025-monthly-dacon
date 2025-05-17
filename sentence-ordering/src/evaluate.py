from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from peft import PeftModel
import pandas as pd
import torch

from src.prompts import prompt_template
from src.data import load_data

IS_CUDA    = 'cuda' if torch.cuda.is_available() else 'cpu'
TEMPLATE   = prompt_template[False]

def merge_n_unload(
                    base_model  : "AutoModelForCausalLM",
                    adapter_path: str
                  ) -> pipeline:
  
  tuned_model = PeftModel.from_pretrained(base_model, adapter_path)
  tuned_model = tuned_model.merge_and_unload()

  tokenizer  = AutoTokenizer.from_pretrained(adapter_path) 
  pipeline_  = pipeline('text-generation', model = tuned_model,
                        tokenizer = tokenizer, max_new_tokens = 256)

  return pipeline_


def generate_result(pipeline, sample):
  
    sample = sample['prompt']
    result = pipeline(sample, 
                      temperature = 0.6,
                      do_sample   = True
                    )

    result = result[0]['generated_text'][len(sample): ]
    result = result.split('<|im_start}>')[0].strip()
    return result


def sample_evaluate(
                      base_model  : "CausalLM",
                      adapter_path: str
                    ):
  test_dataset = load_data('data', phase_type = 'test', **{'has_context' : False})
  submit       = pd.read_csv('data/sample_submission.csv')

  pipeline_    = merge_n_unload(
                                base_model = base_model,
                                adapter_path = adapter_path
                              )
  sample_data  = test_dataset.iloc[0]
  result       = generate_result(pipeline = pipeline_, sample = sample_data)
  
  return result