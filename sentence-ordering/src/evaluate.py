from ast import literal_eval
import re

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

  tokenizer  = AutoTokenizer.from_pretrained(f'{adapter_path}/tokenizer') 
  pipeline_  = pipeline('text-generation', model = tuned_model,
                        tokenizer = tokenizer, max_new_tokens = 256)

  return pipeline_


def generate_result(pipeline, sample):
    result = pipeline(sample, 
                      temperature = 0.2,
                      do_sample   = True
                    )

    result = result[0]['generated_text'][len(sample): ]

    try: result = result.split('assistant')[1].split('\n\n')[0].strip().replace("'", '"')
    except Exception as e: return None
    
    # result = re.sub(r'(?<!\w)"(.*?)"(?!\w)', r"'\1'", result)
    # result = re.sub(r'"([^"]*?)\'', r"'\1'", result)

    return result


def sample_evaluate(
                      base_model  : "CausalLM",
                      adapter_path: str
                    ):
  test_dataset = load_data('data', phase_type = 'valid', **{'has_context' : False})
  raw_dataset  = pd.read_csv(f'data/valid.csv')

  pipeline_    = merge_n_unload(
                                base_model = base_model,
                                adapter_path = adapter_path
                              )

  ground_truths, predictions = [], []
  for sample, row in zip(test_dataset['prompt'], raw_dataset.iterrows()):
    _, row = row

    gt     = ''.join([str(row[f'answer_{idx}']) for idx in range(4)])
    inputs = [row[f'sentence_{idx}'] for idx in range(4)]
    result = generate_result(pipeline_, sample)

    if result:

      print(result)
      try: result = literal_eval(result)
      except: result = literal_eval(result.split('  tool_list')[0].strip())

      try:  pred   = ''.join([str(inputs.index(res[1])) for res in result])
      except: pred = '0123'
      ground_truths.append(gt)
      predictions.append(pred)

      print(gt == pred, f'[gt] {gt}, [pred] {pred}')
      print('==='*50)
    
    else: continue

  print(accuracy_score(ground_truths, predictions))
  print(f1_score(ground_truths, predictions))  
  return result
