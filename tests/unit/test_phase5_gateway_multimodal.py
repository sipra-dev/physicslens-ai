from __future__ import annotations

import unittest
from types import SimpleNamespace

from pydantic import BaseModel

from src.config import settings
from src.models.contracts import ModelTask
from src.models.gateway import LLMGateway
from src.models.routing import ModelRouter


class DummyStructured(BaseModel):
    message: str


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            output_parsed=DummyStructured(
                message="ok"
            ),
            usage=None,
            output=[],
            _request_id="req_test",
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class Phase5GatewayMultimodalTests(
    unittest.TestCase
):
    def _router(self) -> ModelRouter:
        return ModelRouter(
            classifier_model=(
                settings.phase5_classifier_model
            ),
            text_model=settings.tutor_text_model,
            multimodal_model=(
                settings.tutor_multimodal_model
            ),
            reasoning_model=(
                settings.tutor_reasoning_model
            ),
            verifier_model=(
                settings.verifier_model
            ),
            fallback_model=(
                settings.model_fallback_model
            ),
        )

    def _gateway(
        self,
    ) -> tuple[LLMGateway, FakeClient]:
        client = FakeClient()

        gateway = LLMGateway(
            model_router=self._router(),
            client=client,
            timeout_seconds=(
                settings
                .model_gateway_timeout_seconds
            ),
        )

        return gateway, client

    def test_text_call_stays_text_only(
        self,
    ) -> None:
        gateway, client = self._gateway()

        route = self._router().route_task(
            ModelTask.QUERY_SCOPE
        )

        result = gateway.generate_structured(
            route=route,
            system_prompt="System prompt",
            user_prompt="What is force?",
            response_model=DummyStructured,
        )

        self.assertEqual(
            result.message,
            "ok",
        )

        self.assertEqual(
            len(client.responses.calls),
            1,
        )

        request = client.responses.calls[0]

        user_content = (
            request["input"][1]["content"]
        )

        self.assertEqual(
            user_content,
            [
                {
                    "type": "input_text",
                    "text": "What is force?",
                }
            ],
        )

    def test_multimodal_route_sends_image(
        self,
    ) -> None:
        gateway, client = self._gateway()

        route = self._router().route_task(
            ModelTask.TUTOR_MULTIMODAL
        )

        image_url = (
            "data:image/png;base64,AAAA"
        )

        result = gateway.generate_structured(
            route=route,
            system_prompt="System prompt",
            user_prompt="Explain this diagram.",
            response_model=DummyStructured,
            image_urls=[image_url],
        )

        self.assertEqual(
            result.message,
            "ok",
        )

        request = client.responses.calls[0]

        user_content = (
            request["input"][1]["content"]
        )

        self.assertEqual(
            user_content[0],
            {
                "type": "input_text",
                "text": "Explain this diagram.",
            },
        )

        self.assertEqual(
            user_content[1],
            {
                "type": "input_image",
                "image_url": image_url,
            },
        )

    def test_non_vision_route_rejects_image(
        self,
    ) -> None:
        gateway, client = self._gateway()

        route = self._router().route_task(
            ModelTask.QUERY_SCOPE
        )

        with self.assertRaises(ValueError):
            gateway.generate_structured(
                route=route,
                system_prompt="System prompt",
                user_prompt="Test",
                response_model=DummyStructured,
                image_urls=[
                    "data:image/png;base64,AAAA"
                ],
            )

        self.assertEqual(
            len(client.responses.calls),
            0,
        )


if __name__ == "__main__":
    unittest.main()