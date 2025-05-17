import os

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
import torch

os.environ['WANDB_PROJECT'] = 'monthly_dacon_sentence_ordering'
os.environ["WANDB_LOG_MODEL"] = 'output'

def _find_all_linear_names(
                              model: "AutoModelForCausalLM"
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
              r              = 32, 
              bias           = "none",
              task_type      = 'CAUSAL_LM',
              lora_alpha     = 64,
              lora_dropout   = 0.05,
              target_modules = modules,
          )
  
  train_config = SFTConfig(
        fp16                        = True,
        optim                       = 'grokadamw',
        report_to                   = 'wandb',
        num_train_epochs            = 15,
        output_dir                  = 'output',
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
        packing=True,
        per_device_eval_batch_size  = 2,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
    )
  return lora_config, train_config


def prepare_trainer(
                  model: "AutoModelForCausalLM", 
                  dataset: 'datasets.Dataset'
                ):

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

  return trainer