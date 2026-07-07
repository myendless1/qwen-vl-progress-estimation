from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers.utils import ModelOutput


ROBOTWIN_IGNORE_FLOAT = -100.0


@dataclass
class RobotWinOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Any] = None
    attentions: Optional[Any] = None
    robotwin_logits: Optional[Dict[str, torch.Tensor]] = None
    robotwin_progress: Optional[torch.Tensor] = None


class RobotWinRegressionHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        mid = max(1, hidden_size // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        param_dtype = next(self.net.parameters()).dtype
        if hidden_states.dtype != param_dtype:
            hidden_states = hidden_states.to(param_dtype)
        return self.net(hidden_states).squeeze(-1)


def _hidden_size_from_config(config) -> int:
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise ValueError("Cannot infer hidden size from model config.")


class RobotWinQwenWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        done_loss_weight: float = 1.0,
        progress_loss_weight: float = 1.0,
        replan_loss_weight: float = 0.0,
        incident_loss_weight: float = 0.0,
        voting_done: bool = False,
        done_vote_count: int = 5,
    ):
        super().__init__()
        self.base_model = base_model
        hidden_size = _hidden_size_from_config(base_model.config)
        self.voting_done = voting_done
        self.done_vote_count = done_vote_count
        if voting_done:
            self.current_heads = nn.ModuleList(
                RobotWinRegressionHead(hidden_size) for _ in range(done_vote_count)
            )
        else:
            self.current_head = RobotWinRegressionHead(hidden_size)
        self.plan_head = RobotWinRegressionHead(hidden_size)
        self.incident_head = RobotWinRegressionHead(hidden_size)
        self.value_head = RobotWinRegressionHead(hidden_size)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.progress_loss = nn.SmoothL1Loss(beta=0.05, reduction="none")
        self.done_loss_weight = done_loss_weight
        self.progress_loss_weight = progress_loss_weight
        self.replan_loss_weight = replan_loss_weight
        self.incident_loss_weight = incident_loss_weight
        self.config = base_model.config

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            base_model = self.__dict__.get("_modules", {}).get("base_model")
            if base_model is not None and hasattr(base_model, name):
                return getattr(base_model, name)
            raise exc

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            return self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.base_model.resize_token_embeddings(*args, **kwargs)

    def _gather_query_hidden(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = positions.ge(0)
        safe_positions = positions.clamp(min=0)
        batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_idx, safe_positions], valid

    def _masked_bce(self, logits: torch.Tensor, labels: torch.Tensor, valid_pos: torch.Tensor) -> torch.Tensor:
        labels = labels.to(logits.device, dtype=logits.dtype)
        valid = labels.gt(ROBOTWIN_IGNORE_FLOAT / 2) & valid_pos.to(logits.device)
        if not valid.any():
            return logits.sum() * 0.0
        return self.bce(logits[valid], labels[valid]).mean()

    def _masked_progress_loss(self, values: torch.Tensor, labels: torch.Tensor, valid_pos: torch.Tensor) -> torch.Tensor:
        labels = labels.to(values.device, dtype=values.dtype)
        valid = labels.gt(ROBOTWIN_IGNORE_FLOAT / 2) & valid_pos.to(values.device)
        if not valid.any():
            return values.sum() * 0.0
        preds = torch.sigmoid(values[valid])
        return self.progress_loss(preds, labels[valid]).mean()

    def _done_logits(
        self,
        hidden_states: torch.Tensor,
        vote_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.voting_done:
            return self.current_head(hidden_states)
        if vote_indices is None:
            vote_indices = torch.zeros(hidden_states.shape[0], device=hidden_states.device, dtype=torch.long)
        vote_indices = vote_indices.to(hidden_states.device, dtype=torch.long).clamp(0, self.done_vote_count - 1)
        head_dtype = next(self.current_heads[0].parameters()).dtype
        logits = torch.zeros(hidden_states.shape[0], device=hidden_states.device, dtype=head_dtype)
        for idx, head in enumerate(self.current_heads):
            selected = vote_indices.eq(idx)
            if selected.any():
                logits[selected] = head(hidden_states[selected]).to(logits.dtype)
        return logits

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        robotwin_current_query_pos: Optional[torch.Tensor] = None,
        robotwin_plan_query_pos: Optional[torch.Tensor] = None,
        robotwin_incident_query_pos: Optional[torch.Tensor] = None,
        robotwin_value_query_pos: Optional[torch.Tensor] = None,
        robotwin_current_done: Optional[torch.Tensor] = None,
        robotwin_done_vote_index: Optional[torch.Tensor] = None,
        robotwin_need_replan: Optional[torch.Tensor] = None,
        robotwin_incident: Optional[torch.Tensor] = None,
        robotwin_progress: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> RobotWinOutput:
        lm_labels = labels
        if labels is not None and not labels.ne(-100).any():
            lm_labels = None

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=lm_labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        loss = outputs.loss if outputs.loss is not None else outputs.logits.sum() * 0.0

        robotwin_logits = None
        robotwin_progress_pred = None
        if robotwin_current_query_pos is not None:
            last_hidden = outputs.hidden_states[-1]
            current_h, current_valid = self._gather_query_hidden(last_hidden, robotwin_current_query_pos)
            plan_h, plan_valid = self._gather_query_hidden(last_hidden, robotwin_plan_query_pos)
            incident_h, incident_valid = self._gather_query_hidden(last_hidden, robotwin_incident_query_pos)
            value_h, value_valid = self._gather_query_hidden(last_hidden, robotwin_value_query_pos)

            current_logits = self._done_logits(current_h, robotwin_done_vote_index)
            plan_logits = self.plan_head(plan_h)
            incident_logits = self.incident_head(incident_h)
            value_logits = self.value_head(value_h)
            robotwin_progress_pred = torch.sigmoid(value_logits)
            robotwin_logits = {
                "current_done": current_logits,
                "need_replan": plan_logits,
                "incident": incident_logits,
                "progress_raw": value_logits,
            }

            loss = loss + self.done_loss_weight * self._masked_bce(
                current_logits, robotwin_current_done, current_valid
            )
            loss = loss + self.replan_loss_weight * self._masked_bce(
                plan_logits, robotwin_need_replan, plan_valid
            )
            loss = loss + self.incident_loss_weight * self._masked_bce(
                incident_logits, robotwin_incident, incident_valid
            )
            loss = loss + self.progress_loss_weight * self._masked_progress_loss(
                value_logits, robotwin_progress, value_valid
            )

        return RobotWinOutput(
            loss=loss,
            logits=outputs.logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=outputs.hidden_states,
            attentions=getattr(outputs, "attentions", None),
            robotwin_logits=robotwin_logits,
            robotwin_progress=robotwin_progress_pred,
        )
