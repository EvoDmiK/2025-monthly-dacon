from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.evaluate import sample_evaluate
from src.train import prepare_trainer
from src.data import load_data

DATASET_PATH = 'data'
IS_CUDA      = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
model      = AutoModelForCausalLM.from_pretrained(model_name).to(IS_CUDA)
tokenizer  = AutoTokenizer.from_pretrained(model_name)


def train(**kwargs):

    train_dataset = load_data(
                        DATASET_PATH,
                        **kwargs
                    )
    
    print(f'sample prompt : {train_dataset["train"].data[0][0].as_py()}')
    print(f'sample target : {train_dataset["train"].data[1][0].as_py()}')

    trainer = prepare_trainer(model, train_dataset, tokenizer)
    trainer.train()

    trainer.save_model('output/adapter')
    trainer.tokenizer.save_pretrained('output/adapter/tokenizer')


if __name__ == '__main__':

    kwargs = {'has_context' : True}
    train(**kwargs)

    sample_evaluate(base_model = model, adapter_path = 'output/adapter')