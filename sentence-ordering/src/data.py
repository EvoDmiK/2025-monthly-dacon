from typing import Union

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import datasets

from src.prompts import prompt_template, completion_template

def prepare_prompt(
  row        : pd.Series, 
  phase_type : str  = 'train',
  context_per: int  = 3,
  has_context: bool = False
):
  inputs  = [row[f'sentence_{idx}'] for idx in range(4)]

  if phase_type == 'train':
    targets = [[row[f'answer_{idx}'], input_.replace("'", '"')] \
                for idx, input_ in enumerate(inputs)]
    
    targets = sorted(targets, key = lambda x: x[0])


  else: targets = '' 


  if has_context:
      choices     = [False] * (10 - context_per) + [True] * context_per
      has_context = np.random.choice(choices, size = 1)[0]
      template    = prompt_template[has_context]
  
  else:
      template = prompt_template[has_context]


  prompt     = template.format(inputs = inputs)
  completion = completion_template.format(targets = targets)

  return prompt, completion


def load_data(
                dataset_path: str,
                phase_type  : str = 'train',
                **kwargs
            ) -> Union[datasets.Dataset, pd.DataFrame]:
    
    has_context = kwargs.get('has_context', True)
    context_per = kwargs.get('context_per', 3)
    test_size   = kwargs.get('test_size', 0.2)
    seed        = kwargs.get('seed', 42)

    train_dataset_path = f'{dataset_path}/{phase_type}.csv'
    train_df           = pd.read_csv(train_dataset_path)

    dataset = {'prompt' : [], 'completion' : []} 

    for row in train_df.iterrows():
        _     ,        row  = row
        prompt, completion  = prepare_prompt(
                                                row, 
                                                has_context = has_context, 
                                                context_per = context_per,
                                                phase_type  = phase_type
                                            )

        dataset['prompt'].append(prompt)
        dataset['completion'].append(completion)
      
    dataset = pd.DataFrame(dataset)

    if phase_type == 'train':
      dataset      = datasets.Dataset.from_pandas(dataset).shuffle(seed)
      dataset      = dataset.train_test_split(test_size = test_size)

    return dataset


def train_valid_split(
  dataset_path: str    
):
  
  dataset      = pd.read_csv(dataset_path)
  train, valid = train_test_split(dataset, test_size = 0.1)

  train.to_csv('data/train.csv', index = False)
  valid.to_csv('data/valid.csv', index = False)