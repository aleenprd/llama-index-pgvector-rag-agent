import json
import requests
import datetime
from loguru import logger

from typing import Dict, Any, Optional


def get_formatted_timestamp():
    """
    Gets the current timestamp and formats it as YYYY-MM-DD HH:MM:SS.

    Returns:
    str: The formatted timestamp as a string.  Returns "Error" if there's an issue.
    """
    try:
        formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return formatted_time
    except Exception as e:
        print(f"Error: {e}")
        return "Error"


def milliseconds_to_seconds(milliseconds: float) -> float:
    """Convert milliseconds to seconds.
    
    Args:
        milliseconds: Time duration in milliseconds
        
    Returns:
        float: Time duration in seconds, rounded to 3 decimal places
    """
    return round(milliseconds / 1000, 3)


class LlamaCppClient:
    """Client for interacting with llama.cpp server."""

    def __init__(
        self, host: str = "localhost", port: int = 8000, model: Optional[str] = None
    ):
        """Initialize the client.

        Args:
            host: Server host (default: localhost)
            port: Server port (default: 8000)
            model: Model name (optional, will use server default if not specified)
        """
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.session = requests.Session()

        # If no model is specified, try to get it from the server
        if self.model is None:
            self.set_model_from_server()
        logger.critical(f"Using model: {self.model}")

    def check_health(self) -> bool:
        """Check if the server is running and accessible.

        Returns:
            bool: True if the server is healthy, False otherwise.
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_models(self) -> Dict[str, Any]:
        """Get list of available models from the server.

        Returns:
            Dict[str, Any]: Dictionary containing model information.
        """
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            logger.debug(f"Available models: {response.json()}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting models: {e}")
            return {}

    def process_response(self, response: dict, prompt: str = None) -> dict:
        """Process the response from the server.

        Args:
            response (dict): The raw response from the server.
            prompt (str): The original prompt used for the request.

        Returns:
            dict: Processed response containing relevant information.
        """
        created_at = get_formatted_timestamp()
        timings = response["timings"]
        prompt_eval_duration_ms = timings["prompt_ms"]
        prediction_duration_ms = timings["predicted_ms"]
        total_duration_ms = prompt_eval_duration_ms + prediction_duration_ms
        generation_settings = response["generation_settings"]

        return {
            "seed": generation_settings["seed"],
            "temperature": generation_settings["temperature"],
            "prompt": prompt,
            "content": response["content"],
            "created_at": created_at,
            "tokens_evaluated": round(response["tokens_evaluated"], 3),
            "tokens_predicted": response["tokens_predicted"],
            "total_tokens": response["tokens_evaluated"] + response["tokens_predicted"],
            "prompt_eval_duration_s": milliseconds_to_seconds(prompt_eval_duration_ms),
            "prediction_duration_s": milliseconds_to_seconds(prediction_duration_ms),
            "total_duration_s": milliseconds_to_seconds(total_duration_ms),
            "prompt_eval_duration_ms": prompt_eval_duration_ms,
            "prediction_duration_ms": prediction_duration_ms,
            "total_duration_ms": total_duration_ms,
            "eval_tokens_per_s": round(timings["prompt_per_second"], 3),
            "prediction_tokens_per_s": round(timings["predicted_per_second"], 3),
        }

    def complete(
        self,
        prompt: str,
        n_predict: int = 128,  # Number of tokens to predict
        processed: bool = True,
    ) -> dict:
        """
        Use the server's chat completions function.

        Args:
        - prompt (str): the system prompt
        - format (str | dict, optional): this will force the LLM to generate the response in a given format. Defaults to JSON.
        - processed (bool, optional): specifies wether to post-process the response. Defaults to True.

        Note:
        - A more well-defined format can look like this:
        "format": {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
                "name": {"type": "string"},
                },
            "required": ["name", "age"],
        }

        Returns:
        - dict: the response from the LLM (processed or not)
        """
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
        }

        try:
            response = self.session.post(
                f"{self.base_url}/completion",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()

            return (
                response.json()
                if not processed
                else self.process_response(response.json(), prompt)
            )
        except requests.exceptions.RequestException as e:
            print(f"Error sending request: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing response: {e}")
            return None

    def set_model_from_server(self) -> Optional[str]:
        """Fetch and set the model name from the server's available models.

        Returns:
            Optional[str]: The name of the selected model, or None if no models available
        """
        models_response = self.get_models()

        if (
            models_response
            and "models" in models_response
            and models_response["models"]
        ):
            # Get the first model's name
            self.model = models_response["models"][0]["name"]
            return self.model

        if models_response and "data" in models_response and models_response["data"]:
            # Alternative structure for models response
            self.model = models_response["data"][0]["id"]
            return self.model

        logger.error("No models available from the server.")

        return None
