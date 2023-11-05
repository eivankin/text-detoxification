import openai

from src.models.abstract import BaseModel


class ZeroShotModel(BaseModel):
    def __init__(self, api_key: str, model_name: str = "text-davinci-003"):
        openai.api_key = api_key
        self.model_name = model_name
        self.max_tokens = 600
        self.temp = 0.6
        self.single_template = (
            "Paraphrase the following sentence to make it less toxic:\n"
            "```{input}```\nOutput only the result sentence."
        )
        self.multiple_template = (
            "Paraphrase the following sentences to make them less toxic:\n"
            "```{input}```\nOutput only the result sentences separated "
            "by line breaks."
        )

    def _send_req(self, prompt: str) -> list[str]:
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            temperature=self.temp,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
        )
        return response.choices[0].text.strip().split("\n")

    def predict_single(self, input_sentence: str) -> str:
        return self._send_req(self.single_template.format(input=input_sentence))[0]

    def predict(self, inputs: list[str]) -> list[str]:
        all_sentences = "\n".join(inputs)
        return self._send_req(self.multiple_template.format(input=all_sentences))


class FewShotModel(ZeroShotModel):
    def __init__(
        self,
        api_key: str,
        ref_samples: list[str],
        trn_samples: list[str],
        model_name: str = "text-davinci-003",
    ):
        super().__init__(api_key, model_name)
        refs_joined = "\n".join(ref_samples)
        trns_joined = "\n".join(trn_samples)
        self.single_template = (
            "Input sentences:\n"
            + refs_joined
            + "\nParaphrased and less toxic sentences:\n"
            + trns_joined
            + "\nInput sentences: {input}\nParaphrased and less toxic sentences:\n"
        )
        self.multiple_template = self.single_template
