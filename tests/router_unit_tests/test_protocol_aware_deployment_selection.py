import pytest

from litellm.proxy.route_llm_request import _tag_input_protocol
from litellm.router import Router


def test_tag_input_protocol_replaces_non_dict_metadata() -> None:
    data = {"metadata": "bad-metadata"}

    _tag_input_protocol(data, "anthropic_messages")

    assert data["metadata"] == {"litellm_input_protocol": "anthropic"}


def test_get_available_deployment_prefers_matching_protocol() -> None:
    router = Router(
        model_list=[
            {
                "model_name": "shared-model",
                "litellm_params": {"model": "openai/shared-model"},
                "model_info": {"id": "oai"},
            },
            {
                "model_name": "shared-model",
                "litellm_params": {"model": "anthropic/shared-model"},
                "model_info": {"id": "ant"},
            },
        ],
        routing_strategy="simple-shuffle",
    )

    openai_deployment = router.get_available_deployment(
        model="shared-model",
        request_kwargs={"metadata": {"litellm_input_protocol": "openai"}},
    )
    anthropic_deployment = router.get_available_deployment(
        model="shared-model",
        request_kwargs={"metadata": {"litellm_input_protocol": "anthropic"}},
    )

    assert openai_deployment["model_info"]["id"] == "oai"
    assert anthropic_deployment["model_info"]["id"] == "ant"


@pytest.mark.asyncio
async def test_async_get_available_deployment_prefers_matching_protocol() -> None:
    router = Router(
        model_list=[
            {
                "model_name": "shared-model",
                "litellm_params": {"model": "github_copilot/gpt-5.4"},
                "model_info": {"id": "copilot"},
            },
            {
                "model_name": "shared-model",
                "litellm_params": {"model": "anthropic/claude-sonnet-4.6"},
                "model_info": {"id": "anthropic"},
            },
        ],
        routing_strategy="simple-shuffle",
    )

    deployment = await router.async_get_available_deployment(
        model="shared-model",
        request_kwargs={"metadata": {"litellm_input_protocol": "anthropic"}},
    )

    assert deployment["model_info"]["id"] == "anthropic"
