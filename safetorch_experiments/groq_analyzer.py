import json
import os
from groq import Groq

class GroqAnalyzer:
    def __init__(self, api_key, model_name="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.requirements_text = ""

    def load_requirements(self, file_path):
        """Loads all requirements from a single text file."""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.requirements_text = f.read().strip()
            print(f"Requirements loaded from {file_path}")
        else:
            print(f"Warning: Requirements file not found at {file_path}")
            self.requirements_text = "Standard Parser requirements."

    def analyze_assembly(self, assembly_code):
        """
        Sends assembly to Groq using a specific classification prompt.
        Returns a binary prediction and a short reasoning string.
        """
        
        # We use a System Prompt to force the JSON structure 
        # so we can capture BOTH the bit and the reasoning.
        system_prompt = (
            "You are a binary analysis assistant. "
            "Your task is to classify code and provide a VERY brief reasoning. "
            "You MUST return your response in JSON format with exactly two keys: "
            "'prediction' (integer 0 or 1) and 'reasoning' (a string of max 3 lines)."
        )

        # This is your EXACT prompt integrated into the user content
        user_content = (
            "You are a binary analysis assistant.\n"
            "I will give you the code of a function\n"
            "Your task is to classify whether the function is\n"
            "a PARSER function or not based on the following requirements:\n"
            "Classification requirements:\n"
            f"{self.requirements_text}\n\n"
            "Your output MUST follow these strict rules:\n"
            "- Output ONLY a single character:\n"
            "'1' if the function satisfies ALL requirements\n"
            "or at least the final one (Composition).\n"
            "- Output '0' if the function does NOT satisfy the criteria.\n"
            "- NO explanations.\n"
            "- NO extra text.\n"
            "- NO formatting. Only '0' or '1'.\n"
            "Function code:\n"
            f"{assembly_code}\n\n"
            "Prediction (ONLY '0' or '1'):"
        )

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model=self.model_name,
                response_format={"type": "json_object"},
                temperature=0.1, # Keep it deterministic for 0/1 classification
            )
            
            # Parse the JSON response
            resp_content = json.loads(chat_completion.choices[0].message.content)
            
            return {
                "llm_prediction": int(resp_content.get("prediction", 0)),
                "llm_reasoning": str(resp_content.get("reasoning", "No reasoning provided.")).strip()
            }
        
        except Exception as e:
            print(f"Error calling Groq: {e}")
            return {"llm_prediction": 0, "llm_reasoning": "Analysis failed due to connection error."}