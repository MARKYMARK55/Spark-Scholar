from typing import Any

import requests
from langchain_anthropic import ChatAnthropic
from langchain_ibm import ChatWatsonx
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr

from lfx.base.models.anthropic_constants import ANTHROPIC_MODELS
from lfx.base.models.google_generative_ai_constants import GOOGLE_GENERATIVE_AI_MODELS
from lfx.base.models.google_generative_ai_model import ChatGoogleGenerativeAIFixed
from lfx.base.models.model import LCModelComponent
from lfx.base.models.model_utils import get_ollama_models, is_valid_ollama_url
from lfx.base.models.openai_constants import OPENAI_CHAT_MODEL_NAMES, OPENAI_REASONING_MODEL_NAMES
from lfx.field_typing import LanguageModel
from lfx.field_typing.range_spec import RangeSpec
from lfx.inputs.inputs import BoolInput, MessageTextInput, StrInput
from lfx.io import DropdownInput, MessageInput, MultilineInput, SecretStrInput, SliderInput
from lfx.log.logger import logger
from lfx.schema.dotdict import dotdict
from lfx.utils.util import transform_localhost_url

# IBM watsonx.ai constants
IBM_WATSONX_DEFAULT_MODELS = ["ibm/granite-3-2b-instruct", "ibm/granite-3-8b-instruct", "ibm/granite-13b-instruct-v2"]
IBM_WATSONX_URLS = [
    "https://us-south.ml.cloud.ibm.com",
    "https://eu-de.ml.cloud.ibm.com",
    "https://eu-gb.ml.cloud.ibm.com",
    "https://au-syd.ml.cloud.ibm.com",
    "https://jp-tok.ml.cloud.ibm.com",
    "https://ca-tor.ml.cloud.ibm.com",
]

# Ollama API constants
HTTP_STATUS_OK = 200
JSON_MODELS_KEY = "models"
JSON_NAME_KEY = "name"
JSON_CAPABILITIES_KEY = "capabilities"
DESIRED_CAPABILITY = "completion"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# LiteLLM proxy constants
DEFAULT_LITELLM_URL = "http://litellm:4000"
DEFAULT_LITELLM_API_KEY = "simple-api-key"


class LanguageModelComponent(LCModelComponent):
    display_name = "LiteLLM Language Model"
    description = "Runs a language model given a specified provider."
    documentation: str = "https://docs.langflow.org/components-models"
    icon = "brain-circuit"
    category = "models"
    priority = 0

    @staticmethod
    def fetch_ibm_models(base_url: str) -> list[str]:
        """Fetch available models from the watsonx.ai API."""
        try:
            endpoint = f"{base_url}/ml/v1/foundation_model_specs"
            params = {"version": "2024-09-16", "filters": "function_text_chat,!lifecycle_withdrawn"}
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [model["model_id"] for model in data.get("resources", [])]
            return sorted(models)
        except Exception:  # noqa: BLE001
            logger.exception("Error fetching IBM watsonx models. Using default models.")
            return IBM_WATSONX_DEFAULT_MODELS

    @staticmethod
    def fetch_litellm_models(base_url: str, api_key: str) -> list[str]:
        """Fetch available models from the LiteLLM proxy /models endpoint."""
        try:
            url = f"{base_url.rstrip('/')}/v1/models"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return sorted(models) if models else ["Nemotron-Nano", "GPT-OSS"]
        except Exception:  # noqa: BLE001
            logger.exception("Error fetching LiteLLM models. Using defaults from config.")
            return ["Nemotron-Nano", "GPT-OSS"]

    inputs = [
        DropdownInput(
            name="provider",
            display_name="Model Provider",
            options=["LiteLLM (Local)", "OpenAI", "Anthropic", "Google", "IBM watsonx.ai", "Ollama"],
            value="LiteLLM (Local)",
            info="Select the model provider",
            real_time_refresh=True,
            options_metadata=[
                {"icon": "CPU"},           # LiteLLM local
                {"icon": "OpenAI"},
                {"icon": "Anthropic"},
                {"icon": "GoogleGenerativeAI"},
                {"icon": "WatsonxAI"},
                {"icon": "Ollama"},
            ],
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=["Nemotron-Nano", "GPT-OSS"],
            value="GPT-OSS",
            info="Select the model to use",
            real_time_refresh=True,
            refresh_button=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="API key for the selected provider. For LiteLLM local proxy use: simple-api-key",
            required=False,
            show=True,
            value=DEFAULT_LITELLM_API_KEY,
            real_time_refresh=True,
        ),
        # ── LiteLLM-specific inputs ──
        MessageTextInput(
            name="litellm_base_url",
            display_name="LiteLLM Proxy URL",
            info="Base URL of your LiteLLM proxy (without /v1). Default points to the litellm container on llm-net.",
            value=DEFAULT_LITELLM_URL,
            show=True,
            real_time_refresh=True,
            load_from_db=True,
        ),
        # ── IBM watsonx-specific inputs ──
        DropdownInput(
            name="base_url_ibm_watsonx",
            display_name="watsonx API Endpoint",
            info="The base URL of the API (IBM watsonx.ai only)",
            options=IBM_WATSONX_URLS,
            value=IBM_WATSONX_URLS[0],
            show=False,
            real_time_refresh=True,
        ),
        StrInput(
            name="project_id",
            display_name="watsonx Project ID",
            info="The project ID associated with the foundation model (IBM watsonx.ai only)",
            show=False,
            required=False,
        ),
        # ── Ollama-specific inputs ──
        MessageTextInput(
            name="ollama_base_url",
            display_name="Ollama API URL",
            info=f"Endpoint of the Ollama API (Ollama only). Defaults to {DEFAULT_OLLAMA_URL}",
            value=DEFAULT_OLLAMA_URL,
            show=False,
            real_time_refresh=True,
            load_from_db=True,
        ),
        # ── Common inputs ──
        MessageInput(
            name="input_value",
            display_name="Input",
            info="The input text to send to the model",
        ),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            info="A system message that helps set the behavior of the assistant",
            advanced=False,
        ),
        BoolInput(
            name="stream",
            display_name="Stream",
            info="Whether to stream the response",
            value=False,
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            info="Controls randomness in responses",
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
    ]

    def build_model(self) -> LanguageModel:
        provider = self.provider
        model_name = self.model_name
        temperature = self.temperature
        stream = self.stream

        # ── LiteLLM local proxy ──
        if provider == "LiteLLM (Local)":
            base_url = getattr(self, "litellm_base_url", DEFAULT_LITELLM_URL).rstrip("/")
            api_key = self.api_key or DEFAULT_LITELLM_API_KEY
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                api_key=api_key,
                base_url=f"{base_url}/v1",
            )

        # ── OpenAI ──
        if provider == "OpenAI":
            if not self.api_key:
                msg = "OpenAI API key is required when using OpenAI provider"
                raise ValueError(msg)
            if model_name in OPENAI_REASONING_MODEL_NAMES:
                temperature = None
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                api_key=self.api_key,
            )

        # ── Anthropic ──
        if provider == "Anthropic":
            if not self.api_key:
                msg = "Anthropic API key is required when using Anthropic provider"
                raise ValueError(msg)
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                anthropic_api_key=self.api_key,
            )

        # ── Google ──
        if provider == "Google":
            if not self.api_key:
                msg = "Google API key is required when using Google provider"
                raise ValueError(msg)
            return ChatGoogleGenerativeAIFixed(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                google_api_key=self.api_key,
            )

        # ── IBM watsonx.ai ──
        if provider == "IBM watsonx.ai":
            if not self.api_key:
                msg = "IBM API key is required when using IBM watsonx.ai provider"
                raise ValueError(msg)
            if not self.base_url_ibm_watsonx:
                msg = "IBM watsonx API Endpoint is required when using IBM watsonx.ai provider"
                raise ValueError(msg)
            if not self.project_id:
                msg = "IBM watsonx Project ID is required when using IBM watsonx.ai provider"
                raise ValueError(msg)
            return ChatWatsonx(
                apikey=SecretStr(self.api_key).get_secret_value(),
                url=self.base_url_ibm_watsonx,
                project_id=self.project_id,
                model_id=model_name,
                params={"temperature": temperature},
                streaming=stream,
            )

        # ── Ollama ──
        if provider == "Ollama":
            if not self.ollama_base_url:
                msg = "Ollama API URL is required when using Ollama provider"
                raise ValueError(msg)
            if not model_name:
                msg = "Model name is required when using Ollama provider"
                raise ValueError(msg)
            transformed_base_url = transform_localhost_url(self.ollama_base_url)
            if transformed_base_url and transformed_base_url.rstrip("/").endswith("/v1"):
                transformed_base_url = transformed_base_url.rstrip("/").removesuffix("/v1")
                logger.warning(
                    "Detected '/v1' suffix in base URL. The Ollama component uses the native Ollama API, "
                    "not the OpenAI-compatible API. The '/v1' suffix has been automatically removed."
                )
            return ChatOllama(
                base_url=transformed_base_url,
                model=model_name,
                temperature=temperature,
            )

        msg = f"Unknown provider: {provider}"
        raise ValueError(msg)

    async def update_build_config(
        self, build_config: dotdict, field_value: Any, field_name: str | None = None
    ) -> dotdict:

        if field_name == "provider":

            # ── LiteLLM (Local) ──
            if field_value == "LiteLLM (Local)":
                build_config["litellm_base_url"]["show"] = True
                build_config["api_key"]["display_name"] = "LiteLLM API Key"
                build_config["api_key"]["show"] = True
                build_config["base_url_ibm_watsonx"]["show"] = False
                build_config["project_id"]["show"] = False
                build_config["ollama_base_url"]["show"] = False

                # Fetch models from the running proxy
                base_url = getattr(self, "litellm_base_url", DEFAULT_LITELLM_URL)
                api_key = getattr(self, "api_key", DEFAULT_LITELLM_API_KEY) or DEFAULT_LITELLM_API_KEY
                models = self.fetch_litellm_models(base_url=base_url, api_key=api_key)
                build_config["model_name"]["options"] = models
                build_config["model_name"]["value"] = models[0] if models else "GPT-OSS"

            elif field_value == "OpenAI":
                build_config["litellm_base_url"]["show"] = False
                build_config["model_name"]["options"] = OPENAI_CHAT_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES
                build_config["model_name"]["value"] = OPENAI_CHAT_MODEL_NAMES[0]
                build_config["api_key"]["display_name"] = "OpenAI API Key"
                build_config["api_key"]["show"] = True
                build_config["base_url_ibm_watsonx"]["show"] = False
                build_config["project_id"]["show"] = False
                build_config["ollama_base_url"]["show"] = False

            elif field_value == "Anthropic":
                build_config["litellm_base_url"]["show"] = False
                build_config["model_name"]["options"] = ANTHROPIC_MODELS
                build_config["model_name"]["value"] = ANTHROPIC_MODELS[0]
                build_config["api_key"]["display_name"] = "Anthropic API Key"
                build_config["api_key"]["show"] = True
                build_config["base_url_ibm_watsonx"]["show"] = False
                build_config["project_id"]["show"] = False
                build_config["ollama_base_url"]["show"] = False

            elif field_value == "Google":
                build_config["litellm_base_url"]["show"] = False
                build_config["model_name"]["options"] = GOOGLE_GENERATIVE_AI_MODELS
                build_config["model_name"]["value"] = GOOGLE_GENERATIVE_AI_MODELS[0]
                build_config["api_key"]["display_name"] = "Google API Key"
                build_config["api_key"]["show"] = True
                build_config["base_url_ibm_watsonx"]["show"] = False
                build_config["project_id"]["show"] = False
                build_config["ollama_base_url"]["show"] = False

            elif field_value == "IBM watsonx.ai":
                build_config["litellm_base_url"]["show"] = False
                build_config["model_name"]["options"] = IBM_WATSONX_DEFAULT_MODELS
                build_config["model_name"]["value"] = IBM_WATSONX_DEFAULT_MODELS[0]
                build_config["api_key"]["display_name"] = "IBM API Key"
                build_config["api_key"]["show"] = True
                build_config["base_url_ibm_watsonx"]["show"] = True
                build_config["project_id"]["show"] = True
                build_config["ollama_base_url"]["show"] = False

            elif field_value == "Ollama":
                build_config["litellm_base_url"]["show"] = False
                build_config["api_key"]["show"] = False
                build_config["base_url_ibm_watsonx"]["show"] = False
                build_config["project_id"]["show"] = False
                build_config["ollama_base_url"]["show"] = True

                ollama_url = getattr(self, "ollama_base_url", None)
                if not ollama_url:
                    config_value = build_config["ollama_base_url"].get("value", DEFAULT_OLLAMA_URL)
                    is_variable_ref = (
                        config_value
                        and isinstance(config_value, str)
                        and config_value.isupper()
                        and "_" in config_value
                    )
                    ollama_url = DEFAULT_OLLAMA_URL if is_variable_ref else config_value

                if await is_valid_ollama_url(url=ollama_url):
                    try:
                        models = await get_ollama_models(
                            base_url_value=ollama_url,
                            desired_capability=DESIRED_CAPABILITY,
                            json_models_key=JSON_MODELS_KEY,
                            json_name_key=JSON_NAME_KEY,
                            json_capabilities_key=JSON_CAPABILITIES_KEY,
                        )
                        build_config["model_name"]["options"] = models
                        build_config["model_name"]["value"] = models[0] if models else ""
                    except ValueError:
                        build_config["model_name"]["options"] = []
                        build_config["model_name"]["value"] = ""
                else:
                    build_config["model_name"]["options"] = []
                    build_config["model_name"]["value"] = ""

        # ── Refresh LiteLLM models when proxy URL changes ──
        elif field_name == "litellm_base_url":
            if hasattr(self, "provider") and self.provider == "LiteLLM (Local)":
                api_key = getattr(self, "api_key", DEFAULT_LITELLM_API_KEY) or DEFAULT_LITELLM_API_KEY
                models = self.fetch_litellm_models(base_url=field_value, api_key=api_key)
                build_config["model_name"]["options"] = models
                build_config["model_name"]["value"] = models[0] if models else "GPT-OSS"

        elif field_name == "base_url_ibm_watsonx" and field_value and hasattr(self, "provider") and self.provider == "IBM watsonx.ai":
            try:
                models = self.fetch_ibm_models(base_url=field_value)
                build_config["model_name"]["options"] = models
                build_config["model_name"]["value"] = models[0] if models else IBM_WATSONX_DEFAULT_MODELS[0]
            except Exception:  # noqa: BLE001
                logger.exception("Error updating IBM model options.")

        elif field_name == "ollama_base_url":
            if await is_valid_ollama_url(url=self.ollama_base_url):
                try:
                    models = await get_ollama_models(
                        base_url_value=self.ollama_base_url,
                        desired_capability=DESIRED_CAPABILITY,
                        json_models_key=JSON_MODELS_KEY,
                        json_name_key=JSON_NAME_KEY,
                        json_capabilities_key=JSON_CAPABILITIES_KEY,
                    )
                    build_config["model_name"]["options"] = models
                    build_config["model_name"]["value"] = models[0] if models else ""
                except ValueError:
                    build_config["model_name"]["options"] = []
                    build_config["model_name"]["value"] = ""
            else:
                build_config["model_name"]["options"] = []
                build_config["model_name"]["value"] = ""

        elif field_name == "model_name":
            if hasattr(self, "provider") and self.provider == "Ollama":
                ollama_url = getattr(self, "ollama_base_url", DEFAULT_OLLAMA_URL)
                if await is_valid_ollama_url(url=ollama_url):
                    try:
                        models = await get_ollama_models(
                            base_url_value=ollama_url,
                            desired_capability=DESIRED_CAPABILITY,
                            json_models_key=JSON_MODELS_KEY,
                            json_name_key=JSON_NAME_KEY,
                            json_capabilities_key=JSON_CAPABILITIES_KEY,
                        )
                        build_config["model_name"]["options"] = models
                    except ValueError:
                        build_config["model_name"]["options"] = []
                else:
                    build_config["model_name"]["options"] = []

            # Refresh LiteLLM model list on button press
            if hasattr(self, "provider") and self.provider == "LiteLLM (Local)":
                base_url = getattr(self, "litellm_base_url", DEFAULT_LITELLM_URL)
                api_key = getattr(self, "api_key", DEFAULT_LITELLM_API_KEY) or DEFAULT_LITELLM_API_KEY
                models = self.fetch_litellm_models(base_url=base_url, api_key=api_key)
                build_config["model_name"]["options"] = models

            # Hide system_message for o1 models
            if field_value and field_value.startswith("o1") and hasattr(self, "provider") and self.provider == "OpenAI":
                if "system_message" in build_config:
                    build_config["system_message"]["show"] = False
            elif "system_message" in build_config:
                build_config["system_message"]["show"] = True

        return build_config
