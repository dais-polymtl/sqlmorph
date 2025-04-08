import enum
from typing import Union

from src.core.logger import Logger
from src.core.model_manager.ollama_model import (
    OllamaModel,
    OllamaChatCompletion,
    OllamaEmbeddings,
)
from src.core.model_manager.openai_model import (
    OpenAIModel,
    OpenAIChatCompletion,
    OpenAIEmbeddings,
)
from src.core.model_manager.huggingface_model import (
    HuggingFaceModel,
    HuggingFaceChatCompletion,
    HuggingFaceEmbeddings,
)

logger = Logger(__name__)


class ModelProvider(enum.Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class ModelType(enum.Enum):
    COMPLETION = "completion"
    EMBEDDING = "embedding"


class ModelManager:
    @classmethod
    def create_model(
        cls,
        model_provider: ModelProvider,
        model_type: ModelType,
        model_name: Union[OpenAIModel, OllamaModel, HuggingFaceModel],
        openai_api_key: str | None = None,
        portkey_api_key: str | None = None,
        portkey_config_id: str | None = None,
    ):
        if model_provider == ModelProvider.OPENAI:
            if model_type == ModelType.COMPLETION:
                return OpenAIChatCompletion(
                    model_name, openai_api_key, portkey_api_key, portkey_config_id
                )
            elif model_type == ModelType.EMBEDDING:
                return OpenAIEmbeddings(
                    model_name, openai_api_key, portkey_api_key, portkey_config_id
                )
        elif model_provider == ModelProvider.OLLAMA:
            if model_type == ModelType.COMPLETION:
                return OllamaChatCompletion(model_name)
            elif model_type == ModelType.EMBEDDING:
                return OllamaEmbeddings(model_name)
        elif model_provider == ModelProvider.HUGGINGFACE:
            if model_type == ModelType.COMPLETION:
                return HuggingFaceChatCompletion(model_name)
            elif model_type == ModelType.EMBEDDING:
                return HuggingFaceEmbeddings(model_name)
        else:
            logger.log(
                "error",
                "UNSUPPORTED_MODEL_PROVIDER",
                {"MODEL_PROVIDER": model_provider},
            )
            raise ValueError(f"No model found for provider: {model_provider}")
