import enum
import time
from typing import List, Dict, Union

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, pipeline
from src.core.logger import Logger

logger = Logger(__name__)


class HuggingFaceModel(enum.Enum):
    GPT2 = "gpt2", "https://huggingface.co/gpt2"
    BERT_BASE = "bert-base-uncased", "https://huggingface.co/bert-base-uncased"
    ROBERTA_BASE = "roberta-base", "https://huggingface.co/roberta-base"
    QWEN_INSTRUCT = "qwen-instruct", "https://huggingface.co/qwen/Qwen-7B-Instruct"

    @property
    def value(self):
        return self._value_[0]

    def get_model_url(self):
        return self._value_[1]


class HuggingFaceChatCompletion:
    def __init__(self, model_name: HuggingFaceModel):
        self.model_name = model_name.value
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def get_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        **kwargs,
    ):
        try:
            start_time = time.time()
            prompt = "\n".join(message["content"] for message in messages)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                **kwargs,
            )
            completions = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            end_time = time.time()

            num_input_tokens = len(inputs.input_ids[0])
            num_output_tokens = sum(
                len(self.tokenizer.decode(output).split()) for output in outputs
            )

            return {
                "completion_content": completions,
                "completion_latency": end_time - start_time,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
            }
        except Exception as e:
            logger.log(
                "error",
                "CHAT_COMPLETION_FAILED",
                {"HUGGINGFACE_MODEL": self.model_name, "error": str(e)},
            )
            raise


class HuggingFaceEmbeddings:
    def __init__(self, model_name: HuggingFaceModel):
        self.model_name = model_name.value
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def get_embedding(self, input_data: Union[str, List[str]]):
        try:
            if isinstance(input_data, str):
                input_data = [input_data]
            embedding_pipeline = pipeline(
                "feature-extraction", model=self.model, tokenizer=self.tokenizer
            )
            embeddings = embedding_pipeline(input_data, truncation=True)
            return embeddings
        except Exception as e:
            logger.log(
                "error",
                "EMBEDDING_FAILED",
                {
                    "HUGGINGFACE_MODEL": self.model_name,
                    "input_data": input_data,
                    "error": str(e),
                },
            )
            raise
