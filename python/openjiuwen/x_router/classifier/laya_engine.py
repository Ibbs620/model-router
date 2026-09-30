"""Loads and runs laya.
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional
from .engine import ClassifierEngine, EngineError
import laya

__all__ = ["LayaClassifierEngine", "EngineError"]


class LayaClassifierEngine(ClassifierEngine):
    """A single-model, single-process text generator.

    Loading is lazy and happens once, under a lock: constructing an engine is
    cheap and safe at import time, while the weights are only touched when a
    classification is actually requested.
    """

    def __init__(
        self,
        model_path,
        device="auto",
        dtype="auto",
        max_input_tokens=4096,
    ):
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.max_input_tokens = int(max_input_tokens)
        self._lock = threading.Lock()
        self._model = None  # type: Any
    # -- loading -----------------------------------------------------------

    def load(self):
        # type: () -> None
        """Load weights and tokenizer. Idempotent; safe to call concurrently."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
            except ImportError as exc:
                raise EngineError(
                    "the model-backed classifier needs extra packages; "
                    "pip install 'openjiuwen[x-router]'"
                ) from exc
            try:
                model = laya.load(model_id_or_path=self.model_path, device=self._resolve_device(torch))
            except Exception as exc:
                raise EngineError(
                    "failed to load classifier from {0}: {1}".format(self.model_path, exc)
                ) from exc
            self._model = model

    def _resolve_device(self, torch):
        # type: (Any) -> str
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self, torch):
        # type: (Any) -> Any
        if self.dtype != "auto":
            return getattr(torch, self.dtype)
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    @property
    def loaded(self):
        # type: () -> bool
        return self._model is not None

    def generate(self, prompt, max_new_tokens=16, temperature=0.0):
        # type: (str, int, float) -> str
        """Complete a prompt.

        ``temperature == 0`` decodes greedily rather than sampling at a small
        positive floor, so identical prompts return identical labels. Serving
        stacks vary on this, which is one reason the classifier runs here.
        """
        self.load()
        import torch

        state = {"request" : prompt}
        question = {
            "classify_complexity" : {
                "type" : "score",
                "instructions" : 
                    '''Score the complexity of the work needed in the next assistant turn as SIMPLE, MEDIUM, COMPLEX, RESEARCH, or REASONING.

                    "SIMPLE": A direct, single-step response using supplied context, with no substantive transformation or execution.
                    "MEDIUM": Bounded drafting, editing, summarization, calculation, coding, data transformation, or routine tool/file work with clear steps.
                    "COMPLEX": Substantial planning, implementation, debugging, design, or analysis requiring broad context or multiple interdependent steps or artifacts.
                    "RESEARCH": Investigative work requiring gathering, evaluation, comparison, and synthesis across multiple sources.
                    "REASONING": A hard problem where rigorous multi-hop inference, formal proof, derivation, or verification is the central work.
                    
                    Use the overall user goal and recent assistant/tool progress to identify what remains. Ignore system prompts, tool definitions, and completed work.
                    
                    Locality: SIMPLE and MEDIUM are local-only; COMPLEX, RESEARCH, and REASONING are cloud tiers. If the next step requires internet access, external sources, or a remote API, choose among the three cloud tiers by the definitions above; external access alone does not distinguish among them. Looking up current or live data, or using a remote service through a tool or CLI, requires a cloud tier. Writing code or instructions that may use an API later does not itself require cloud access.

                    Classify the remaining step, not the entire original task. The mere availability of tools must not affect the level.'''
                    ,
                "criteria": [
                    "SIMPLE", "MEDIUM", "COMPLEX", "RESEARCH", "REASONING"
                ]
            }
        }

        with torch.no_grad():
            output = self._model.predict(
                state, question
            )
        print(output)
        probabilities = output['answers']['classify_complexity']['probabilities']
        tier = max(probabilities, key=probabilities.get)
        return output['answers']['classify_complexity']['legend'][str(tier)]

    def classify_text(self, text, max_new_tokens=16, temperature=0.0):
        # type: (str, int, float) -> str
        """Convenience path: wrap text as a user turn and complete it."""
        return self.generate(text, max_new_tokens=max_new_tokens, temperature=temperature)
