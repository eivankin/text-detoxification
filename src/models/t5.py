import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.models.abstract import BaseModel
from src.util import get_device


class T5Model(BaseModel):
    def predict_single(self, input_sentence: str) -> str:
        encoding = self.tokenizer.encode_plus(input_sentence, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding[
            "attention_mask"].to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=False,
            num_beams=100,
            max_new_tokens=100
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def __init__(self, model_name: str = "s-nlp/t5-paranmt-detox", device: str = get_device()):
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

if __name__ == '__main__':
    # TODO: cli interface
    ...