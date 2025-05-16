from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
import pandas as pd
import numpy as np
import datasets
import torch

IS_CUDA = 'cuda' if torch.cuda.is_available() else 'cpu'


model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
model      = AutoModelForCausalLM.from_pretrained(model_name).to(IS_CUDA)
tokenizer  = AutoTokenizer.from_pretrained(model_name) 

dataset_path       = 'data'
train_dataset_path = f'{dataset_path}/train.csv'
test_dataset_path  = f'{dataset_path}/test.csv'
submission         = f'{dataset_path}/sample_submission.csv'


train_df = pd.read_csv(train_dataset_path)
test_df  = pd.read_csv(test_dataset_path)
train_df

prompt_template = {
True :
  '''
  <|im_start|>tool_list
  <|im_end}|>
  <|im_start|>system
  당신은 지금부터 문장 분석 전문가 입니다.
  주어진 문장 목록을 보고 논리적으로 문장이 완성되도록 문장을 순서대로 정렬해주세요.
  
  ## 힌트
  - 입력 예시
  [
    '특히 여름철 번식시기에 이소과정에서 어미가 새끼를 데리고 열을 맞추어 종종거리며 이동하는 모습이 많이 회자되는데,',
    '서식지를 벗어나서 인도나 차도를 돌아다니는 경우도 흔하기 때문에 일본에서는 흰뺨검둥오리 모자 행렬이',
    '야생오리이면서 주변에서 흔히 찾아볼 수 있어 매체에서 자주 묘사되기도 한다.',
    '도로를 가로지르며 작은 민폐(?)를 주는 귀여운 모습은 일상 동물 이미지에서 거의 클리셰일 정도.'
  ]
  
  - 출력 예시
  [
    [1, '특히 여름철 번식시기에 이소과정에서 어미가 새끼를 데리고 열을 맞추어 종종거리며 이동하는 모습이 많이 회자되는데,'],
    [2, '서식지를 벗어나서 인도나 차도를 돌아다니는 경우도 흔하기 때문에 일본에서는 흰뺨검둥오리 모자 행렬이'],
    [0, '야생오리이면서 주변에서 흔히 찾아볼 수 있어 매체에서 자주 묘사되기도 한다.'],
    [3, '도로를 가로지르며 작은 민폐(?)를 주는 귀여운 모습은 일상 동물 이미지에서 거의 클리셰일 정도.']
  ]
  <|im_end|>
  <|im_start|>user
  {inputs}
  <|im_end|>
''',
False : 
'''
  <|im_start>tool_list
  <|im_end|>
  <|im_start|>system
  당신은 지금부터 문장 분석 전문가 입니다.
  주어진 문장 목록을 보고 논리적으로 문장이 완성되도록 문장을 순서대로 정렬해주세요.
  <|im_end|>
  <|im_start|>user
  {inputs}
  <|im_end|>
'''
}

comlpetion_template = '''
  <|im_start|>assistant
  {targets}
  <|im_end|>
'''

def prepare_dataset(
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
  completion = comlpetion_template.format(targets = targets)

  return prompt, completion

dataset = {'prompt' : [], 'completion' : []}
for row in train_df.iterrows():
    _, row   = row
    prompt, completion  = prepare_dataset(row)
    dataset['prompt'].append(prompt)
    dataset['completion'].append(completion)

dataset = pd.DataFrame(dataset)
dataset

dataset      = datasets.Dataset.from_pandas(dataset).shuffle(42)
dataset      = dataset.train_test_split(test_size = 0.3)
dataset


def _find_all_linear_names(
                              model: AutoModelForCausalLM
                            ):    
  lora_module_names = set()
  for name, module in model.named_modules():
      if isinstance(module, torch.nn.Linear):
          names = name.split(".")
          lora_module_names.add(names[0] if len(names) == 1 else names[-1])
      if "lm_head" in lora_module_names:  # needed for 16-bit
          lora_module_names.remove("lm_head")

  return list(lora_module_names)


def prepare_config(
                       modules
                    ):
  #* MoRA 타입 6으로 하면 에러나는 부분 수정
  lora_config = LoraConfig(
              r              = 64, 
              bias           = "none",
              task_type      = 'CAUSAL_LM',
              lora_alpha     = 32,
              lora_dropout   = 0.05,
              target_modules = modules,
          )
  train_config = SFTConfig(
        fp16                        = True,
        optim                       = 'grokadamw',
        report_to                   = 'tensorboard',
        num_train_epochs            = 30,
        output_dir                  = f'output',
        eval_steps                  = 5,
        save_steps                  = 10, 
        push_to_hub                 = False,
        weight_decay                = 0.05,
        learning_rate               = 3e-5,
        eval_strategy               = 'epoch',
        logging_steps               = 5,
        save_strategy               = 'epoch',
        dataset_text_field          = 'prompt', 
        lr_scheduler_type           = 'linear',
        load_best_model_at_end      = True,
        per_device_eval_batch_size  = 2,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
    )
  return lora_config, train_config


model.gradient_checkpointing_enable()
model.config.use_cache = False
model   = prepare_model_for_kbit_training(model)
modules = _find_all_linear_names(model)

lora_config, train_config = prepare_config(modules)
model                     = get_peft_model(model, lora_config)

callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]

trainer = SFTTrainer(
    model              = model, 
    train_dataset      = dataset['train'],
    eval_dataset       = dataset['test'],
    args               = train_config,
    peft_config        = lora_config,
    callbacks          = callbacks
)

trainer.train()