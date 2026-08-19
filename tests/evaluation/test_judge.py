import json
from unittest.mock import MagicMock

import pytest

from socratic_hint.evaluation.judge import JUDGE_DIMENSIONS, PedagogicalQualityJudge


class FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class FakeResponse:
    def __init__(self, text: str):
        self.content = [FakeTextBlock(text)]


def make_fake_client(payload: dict) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = FakeResponse(json.dumps(payload))
    return client


def test_score_parses_all_dimensions():
    payload = {
        "scaffolding_vs_telling": 4,
        "correctness": 5,
        "appropriateness": 3,
        "rationale": "Good scaffolding hint.",
    }
    client = make_fake_client(payload)
    judge = PedagogicalQualityJudge(client=client)

    score = judge.score("Solve 2+2.", "Tutor: hi", "Try again.")

    for dim in JUDGE_DIMENSIONS:
        assert score.scores[dim] == payload[dim]
    assert score.rationale == "Good scaffolding hint."


def test_score_raises_on_malformed_json():
    client = MagicMock()
    client.messages.create.return_value = FakeResponse("not json at all")
    judge = PedagogicalQualityJudge(client=client)

    with pytest.raises(json.JSONDecodeError):
        judge.score("problem", "context", "hint")
