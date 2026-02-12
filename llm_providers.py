"""LLM provider integration with fallback support (OpenAI + Cohere)."""

import time
import tiktoken
import cohere
from openai import OpenAI
from config import Config

# Pricing (per million tokens)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "command-r-08-2024": {"input": 3.00, "output": 15.00}
}


class CostTracker:
    """Track API costs."""

    def __init__(self):
        self.total_cost = 0.0
        self.requests = []

    def track_request(self, provider, model, input_tokens, output_tokens):
        pricing = PRICING.get(model, {"input": 3.0, "output": 15.0})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        cost = input_cost + output_cost

        self.total_cost += cost

        self.requests.append({
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        })

        return cost

    def get_summary(self):
        total_input = sum(r["input_tokens"] for r in self.requests)
        total_output = sum(r["output_tokens"] for r in self.requests)

        return {
            "total_requests": len(self.requests),
            "total_cost": self.total_cost,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "average_cost": self.total_cost / max(len(self.requests), 1)
        }

    def check_budget(self, daily_budget):
        if self.total_cost >= daily_budget:
            raise Exception(
                f"Daily budget of ${daily_budget:.2f} exceeded! Current: ${self.total_cost:.2f}"
            )

        percent_used = (self.total_cost / daily_budget) * 100
        if percent_used >= 90:
            print(f"⚠️ Warning: {percent_used:.1f}% of daily budget used")


def count_tokens(text, model="gpt-4o-mini"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        return len(text) // 4


class LLMProviders:
    """Manage OpenAI + Cohere with fallback."""

    def __init__(self):
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.cohere_client = cohere.Client(Config.COHERE_API_KEY)
        self.cost_tracker = CostTracker()

        # Rate limiting
        self.openai_last_call = 0
        self.cohere_last_call = 0

        self.openai_interval = 60.0 / Config.OPENAI_RPM
        self.cohere_interval = 60.0 / Config.COHERE_RPM

    # -------------------------
    # Rate limit helpers
    # -------------------------

    def _wait_openai(self):
        elapsed = time.time() - self.openai_last_call
        if elapsed < self.openai_interval:
            time.sleep(self.openai_interval - elapsed)
        self.openai_last_call = time.time()

    def _wait_cohere(self):
        elapsed = time.time() - self.cohere_last_call
        if elapsed < self.cohere_interval:
            time.sleep(self.cohere_interval - elapsed)
        self.cohere_last_call = time.time()

    # -------------------------
    # OpenAI
    # -------------------------

    def ask_openai(self, prompt, model=None):
        if model is None:
            model = Config.OPENAI_MODEL

        self._wait_openai()

        input_tokens = count_tokens(prompt, model)

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        output_text = response.choices[0].message.content
        output_tokens = count_tokens(output_text, model)

        self.cost_tracker.track_request("openai", model, input_tokens, output_tokens)
        self.cost_tracker.check_budget(Config.DAILY_BUDGET)

        return output_text

    # -------------------------
    # Cohere
    # -------------------------

    def ask_cohere(self, prompt, model="command-r-08-2024"):
        self._wait_cohere()

        input_tokens = len(prompt) // 4

        response = self.cohere_client.chat(
            model=model,
            message=prompt,
            max_tokens=1024
        )

        output_text = response.text
        output_tokens = len(output_text) // 4

        self.cost_tracker.track_request("cohere", model, input_tokens, output_tokens)
        self.cost_tracker.check_budget(Config.DAILY_BUDGET)

        return output_text

    # ⚠️ COMPATIBILITÀ CON I TEST
    def ask_anthropic(self, prompt, model=None):
        """
        Alias per compatibilità con i test.
        Internamente usa Cohere.
        """
        return self.ask_cohere(prompt)

    # -------------------------
    # Fallback
    # -------------------------

    def ask_with_fallback(self, prompt, primary="openai"):
        try:
            if primary == "openai":
                print("Trying OpenAI (primary)...")
                response = self.ask_openai(prompt)
                return {"provider": "openai", "response": response}
            else:
                print("Trying Cohere (primary)...")
                response = self.ask_cohere(prompt)
                return {"provider": "cohere", "response": response}

        except Exception as e:
            print(f"✗ Primary provider failed: {e}")
            print("Falling back to secondary provider...")

            try:
                if primary == "openai":
                    response = self.ask_cohere(prompt)
                    return {"provider": "cohere", "response": response}
                else:
                    response = self.ask_openai(prompt)
                    return {"provider": "openai", "response": response}

            except Exception as e2:
                print(f"✗ Secondary provider also failed: {e2}")
                raise Exception("All providers failed")


# -------------------------
# Manual test
# -------------------------

if __name__ == "__main__":
    providers = LLMProviders()

    print("Testing OpenAI:")
    print(providers.ask_openai("What is Python? Answer in one sentence."))

    print("\nTesting Cohere:")
    print(providers.ask_cohere("What is Python? Answer in one sentence."))

    print("\nTesting fallback:")
    result = providers.ask_with_fallback(
        "What is machine learning? Answer in one sentence."
    )
    print(result)

    summary = providers.cost_tracker.get_summary()
    print("\nTotal cost:", summary["total_cost"])
    print("Total requests:", summary["total_requests"])