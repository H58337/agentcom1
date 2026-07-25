import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenLocal:
    """
    Local inference wrapper for Qwen3-8B (Transformers).
    - Load once in __init__
    - Use chat() like an API call
    """

    def __init__(self, model_path: str, tokenizer_path: str = None):
        """
        Args:
            model_path: local path or HF repo id, e.g. "/data/models/Qwen3-8B" or "Qwen/Qwen3-8B"
            tokenizer_path: local path or HF repo id (default: same as model_path)
        """
        self.model_path = model_path

        # device & dtype defaults
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            major = torch.cuda.get_device_capability()[0]
            # Ampere+ -> bf16 (more stable), else fp16
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        # load tokenizer & model once
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        model_kwargs = dict(trust_remote_code=True, torch_dtype=self.dtype)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        if self.device == "cuda":
            self.model.to(self.device)
        self.model.eval()

        # default generation config (you can change via set_generation_config)
        self.gen_cfg = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
        }

    def set_generation_config(
        self,
        max_new_tokens: int = None,
        do_sample: bool = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
    ):
        """Optional: update default generation parameters."""
        if max_new_tokens is not None:
            self.gen_cfg["max_new_tokens"] = int(max_new_tokens)
        if do_sample is not None:
            self.gen_cfg["do_sample"] = bool(do_sample)
        if temperature is not None:
            self.gen_cfg["temperature"] = float(temperature)
        if top_p is not None:
            self.gen_cfg["top_p"] = float(top_p)
        if top_k is not None:
            self.gen_cfg["top_k"] = int(top_k)

    @torch.inference_mode()
    def chat(self, system_prompt: str, user_prompt: str) -> str | None:
        """
        API-like call:
            resp = bot.chat(system_prompt, user_prompt)
        """
        try:
            print(f"[QwenLocal] Received chat request. System prompt length: {len(system_prompt)}, User prompt length: {len(user_prompt)}")
            messages = [
                {"role": "system", "content": system_prompt+" /no_think"},
                {"role": "user", "content": user_prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )

            inputs = self.tokenizer(text, return_tensors="pt")
            input_device = self.device
            if hasattr(self.model, "device"):
                try:
                    input_device = str(self.model.device)
                except Exception:
                    input_device = self.device
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            out = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_cfg["max_new_tokens"],
                do_sample=self.gen_cfg["do_sample"],
                temperature=self.gen_cfg["temperature"],
                top_p=self.gen_cfg["top_p"],
                top_k=self.gen_cfg["top_k"],
            )

            gen_ids = out[0][inputs["input_ids"].shape[-1]:]
            return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        except Exception as e:
            print("[warning] QwenLocal.chat exception:", repr(e))
            return None

