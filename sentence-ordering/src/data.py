import pandas as pd
import numpy as np
import datasets

from src.prompts import prompt_template, completion_template

def prepare_prompt(
  row        : pd.Series, 
  has_context: bool = False,
  context_per: int  = 3
):
  inputs  = [row[f'sentence_{idx}'] for idx in range(4)]
  targets = [[row[f'answer_{idx}'], input_] \
              for idx, input_ in enumerate(inputs)]

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
            ) -> datasets.Dataset:
    
    has_context = kwargs.get('has_context', True)
    context_per = kwargs.get('context_per', 3)
    test_size   = kwargs.get('test_size', 0.2)
    seed        = kwargs.get('seed', 42)

    train_dataset_path = f'{dataset_path}/{phase_type}.csv'
    train_df           = pd.read_csv(train_dataset_path)

    dataset = {'prompt' : [], 'completion' : []}
    for row in train_df.iterrows():
        _, row   = row
        prompt, completion  = prepare_prompt(
                                                row, 
                                                has_context = has_context, 
                                                context_per = context_per
                                            )
        dataset['prompt'].append(prompt)
        dataset['completion'].append(completion)

    dataset = pd.DataFrame(dataset)
    dataset      = datasets.Dataset.from_pandas(dataset).shuffle(seed)
    dataset      = dataset.train_test_split(test_size = test_size)

    return dataset

