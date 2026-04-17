import logging
import os
from collections.abc import AsyncIterable
from typing import Any, Literal, Annotated
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from agent_framework import (
    AgentSession,
    ChatContext,
    Agent,
    tool,
)
from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions

logger = logging.getLogger(__name__)
load_dotenv()


def get_chat_client() -> OpenAIChatClient:
    """Build an Agent Framework chat client from workshop env vars."""
    model = os.getenv('gpt_deployment') or os.getenv('AZURE_OPENAI_CHAT_MODEL') or os.getenv('AZURE_OPENAI_MODEL')
    azure_endpoint = os.getenv('gpt_endpoint') or os.getenv('AZURE_OPENAI_ENDPOINT')
    api_version = os.getenv('gpt_api_version') or os.getenv('AZURE_OPENAI_API_VERSION')
    api_key = os.getenv('AZURE_OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')

    if not model:
        raise ValueError(
            'Missing model configuration. Set gpt_deployment or AZURE_OPENAI_CHAT_MODEL.'
        )

    if not azure_endpoint:
        raise ValueError(
            'Missing endpoint configuration. Set gpt_endpoint or AZURE_OPENAI_ENDPOINT.'
        )

    client_args: dict[str, Any] = {
        'model': model,
        'azure_endpoint': azure_endpoint,
    }
    if api_version:
        client_args['api_version'] = api_version

    if api_key:
        client_args['api_key'] = api_key
    else:
        client_args['credential'] = DefaultAzureCredential()

    return OpenAIChatClient(**client_args)

# region Response Format


class ResponseFormat(BaseModel):
    """A Response Format model to direct how the model should respond."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


# endregion
# region Agent Framework Agent


class AgentFrameworkProductManagementAgent:
    """Wraps Microsoft Agent Framework-based agents to handle Zava product management tasks."""

    agent: Agent
    session: AgentSession = None
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        # Configure the chat completion service explicitly
        chat_service = get_chat_client()

        # Define the main ProductManagerAgent to delegate tasks to the appropriate agents
        self.agent = Agent(
            client=chat_service,
            name='ProductManagerAgent',
            instructions=(
                "Your role is to carefully analyze the user's request and respond as best as you can. "
                'Your primary goal is precise and efficient delegation to ensure customers and employees receive accurate and specialized '
                'assistance promptly.\n\n'
                'IMPORTANT: You must ALWAYS respond with a valid JSON object in the following format:\n'
                '{"status": "<status>", "message": "<your response>"}\n\n'
                'Where status is one of: "input_required", "completed", or "error".\n'
                '- Use "input_required" when you need more information from the user.\n'
                '- Use "completed" when the task is finished.\n'
                '- Use "error" when something went wrong.\n\n'
                'Never respond with plain text. Always use the JSON format above.'
            ),
            tools=[],
        )

    async def invoke(self, user_input: str, session_id: str) -> dict[str, Any]:
        """Handle synchronous tasks (like tasks/send).

        Args:
            user_input (str): User input message.
            session_id (str): Unique identifier for the session.

        Returns:
            dict: A dictionary containing the content, task completion status,
            and user input requirement.
        """
        await self._ensure_session_exists(session_id)

        # Use Agent Framework's run for a single shot
        response = await self.agent.run(
            messages=user_input,
            session=self.session,
            options=OpenAIChatOptions(response_format=ResponseFormat),
        )
        return self._get_agent_response(response.text)

    async def stream(
        self,
        user_input: str,
        session_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """For streaming tasks we yield the Agent Framework agent's run_stream progress.

        Args:
            user_input (str): User input message.
            session_id (str): Unique identifier for the session.

        Yields:
            dict: A dictionary containing the content, task completion status,
            and user input requirement.
        """
        await self._ensure_session_exists(session_id)

        # text_notice_seen = False
        chunks: list[ChatContext] = []

        async for chunk in self.agent.run_stream(
            messages=user_input,
            session=self.session,
        ):
            if chunk.text:
                chunks.append(chunk.text)

        if chunks:
            yield self._get_agent_response(sum(chunks[1:], chunks[0]))

    def _get_agent_response(
        self, message: ChatContext
    ) -> dict[str, Any]:
        """Extracts the structured response from the agent's message content.

        Args:
            message (ChatContext): The message content from the agent.

        Returns:
            dict: A dictionary containing the content, task completion status, and user input requirement.
        """
        structured_response = None
        try:
            structured_response = ResponseFormat.model_validate_json(
                message
            )
        except ValidationError as e:
            logger.info('Message did not come in JSON format.')
            default_response = {
                'is_task_complete': True,
                'require_user_input': False,
                'content': message
            }
        except:
            logger.error('An unexpected error occurred while processing the message.')
            default_response = {
                'is_task_complete': False,
                'require_user_input': True,
                'content': 'We are unable to process your request at the moment. Please try again.',
            }

        if structured_response and isinstance(structured_response, ResponseFormat):
            response_map = {
                'input_required': {
                    'is_task_complete': False,
                    'require_user_input': True,
                },
                'error': {
                    'is_task_complete': False,
                    'require_user_input': True,
                },
                'completed': {
                    'is_task_complete': True,
                    'require_user_input': False,
                },
            }

            response = response_map.get(structured_response.status)
            if response:
                return {**response, 'content': structured_response.message}

        return default_response

    async def _ensure_session_exists(self, session_id: str) -> None:
        """Ensure the session exists for the given session ID.

        Args:
            session_id (str): Unique identifier for the session.
        """
        if self.session is None or self.session.service_session_id != session_id:
            self.session = self.agent.create_session(session_id=session_id)


# endregion
