import os
from dotenv import load_dotenv
from openai import OpenAI

from marvin.config import ASSISTANT_NAME, DEEPSEEK_MODEL, MAX_HISTORY


SYSTEM_PROMPT = f"""
You are {ASSISTANT_NAME}, a helpful desktop assistant for everyday computer-side questions.
Be concise, practical, and friendly.
Do not claim to control the user's computer unless the app has explicitly provided that capability.
Protect private information and refuse requests that would enable harm, credential theft, malware, or unsafe behavior.
When refusing, give a brief reason and offer a safe alternative when possible.
Use plain text unless the user asks for another format.
Do not use markdown formatting in normal replies.
Do not use **bold**, bullet-heavy formatting, headings, code fences, or decorative symbols unless the user specifically asks for code or structured formatting.
Prefer plain natural sentences because responses may be spoken aloud.
"""


class Brain:
    def __init__(self):
        self.client = None
        self.history = []
        self.model = DEEPSEEK_MODEL
        self.is_demo_mode = False
        self.demo_reason = None
        self.setup_message = None
        self.config_error = None

    def setup(self):
        load_dotenv()
        self.model = os.getenv("DEEPSEEK_MODEL", DEEPSEEK_MODEL)
        demo_requested = _env_bool("MARVIN_DEMO_MODE", False)
        api_key = os.getenv("DEEPSEEK_API_KEY")

        if demo_requested:
            self.is_demo_mode = True
            self.demo_reason = "MARVIN_DEMO_MODE is enabled."
            self.setup_message = (
                "Demo mode is active. Marvin will use sample responses and will not call DeepSeek."
            )
            self.client = None
            return

        if not api_key:
            self.config_error = (
                "Missing DEEPSEEK_API_KEY. Add your API key to .env or enable MARVIN_DEMO_MODE=1."
            )
            self.setup_message = self.config_error
            self.client = None
            return

        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, user_text):
        self.history.append({"role": "user", "content": user_text})
        if self.is_demo_mode:
            reply = self._demo_reply(user_text)
            self.history.append({"role": "assistant", "content": reply})
            return reply

        if self.config_error or self.client is None:
            reply = self.config_error or (
                "Live AI is not configured. Add DEEPSEEK_API_KEY to .env or set MARVIN_DEMO_MODE=1."
            )
            self.history.append({"role": "assistant", "content": reply})
            return reply

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-MAX_HISTORY:]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = (
                "I could not reach the live language model. "
                f"Check your API key, network connection, and model setting. Details: {e}"
            )

        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _demo_reply(self, user_text):
        message = user_text.strip() or "(empty message)"
        return (
            "Demo mode is active, so I am not calling the live LLM. "
            f"I received your message: {message}. "
            "Add DEEPSEEK_API_KEY to .env to enable live responses."
        )


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
