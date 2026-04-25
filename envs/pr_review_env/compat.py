from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server import Action, MCPEnvironment, Observation, State
    from openenv.core.env_server import create_fastapi_app
    from openenv.core.env_server.mcp_types import (
        CallToolAction,
        CallToolObservation,
        ListToolsAction,
        ListToolsObservation,
        Tool,
        ToolError,
        ToolErrorType,
    )

    OPENENV_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when OpenEnv imports fail
    OPENENV_AVAILABLE = False

    ObsT = TypeVar("ObsT")
    ActT = TypeVar("ActT", bound="Action")
    StateT = TypeVar("StateT", bound="State")

    @dataclass
    class StepResult(Generic[ObsT]):
        observation: ObsT
        reward: Optional[float] = None
        done: bool = False

    class Action(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        done: bool = False
        reward: float | None = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        model_config = ConfigDict(extra="allow", validate_assignment=True)
        episode_id: Optional[str] = None
        step_count: int = 0

    class Tool(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str
        description: str = ""
        input_schema: Dict[str, Any] = Field(default_factory=dict)

    class ToolErrorType(str):
        EXECUTION_ERROR = "execution_error"
        TOOL_NOT_FOUND = "tool_not_found"
        INVALID_ARGS = "invalid_args"
        TIMEOUT = "timeout"

    class ToolError(BaseModel):
        model_config = ConfigDict(extra="forbid")
        error_type: str
        message: str

    class ListToolsAction(Action):
        type: str = "list_tools"

    class CallToolAction(Action):
        type: str = "call_tool"
        tool_name: str
        arguments: Dict[str, Any] = Field(default_factory=dict)

    class ListToolsObservation(Observation):
        tools: list[Tool] = Field(default_factory=list)

    class CallToolObservation(Observation):
        tool_name: str
        result: Any = None
        error: Optional[ToolError] = None

    class MCPEnvironment:
        SUPPORTS_CONCURRENT_SESSIONS = False

        def __init__(self, mcp_server: Any | None = None, transform: Any | None = None):
            self.mcp_server = mcp_server
            self.transform = transform

        def _get_tool_registry(self) -> Dict[str, Callable[..., Any]]:
            registry = getattr(self, "_tool_registry", None)
            if registry:
                return registry

            if self.mcp_server is not None and hasattr(self.mcp_server, "get_tools"):
                tools = self.mcp_server.get_tools()
                if asyncio.iscoroutine(tools):
                    tools = asyncio.run(tools)
                return {
                    name: getattr(tool, "fn", tool)
                    for name, tool in tools.items()
                }
            return {}

        def step(
            self,
            action: Action,
            timeout_s: Optional[float] = None,
            **kwargs: Any,
        ) -> Observation:
            if isinstance(action, ListToolsAction):
                tools = []
                for name, func in self._get_tool_registry().items():
                    tools.append(
                        Tool(
                            name=name,
                            description=(func.__doc__ or "").strip(),
                            input_schema={},
                        )
                    )
                return ListToolsObservation(tools=tools)

            if isinstance(action, CallToolAction):
                func = self._get_tool_registry().get(action.tool_name)
                if func is None:
                    return CallToolObservation(
                        tool_name=action.tool_name,
                        error=ToolError(
                            error_type=ToolErrorType.TOOL_NOT_FOUND,
                            message=f"Unknown tool: {action.tool_name}",
                        ),
                    )

                try:
                    result = func(**action.arguments)
                except TypeError as exc:
                    return CallToolObservation(
                        tool_name=action.tool_name,
                        error=ToolError(
                            error_type=ToolErrorType.INVALID_ARGS,
                            message=str(exc),
                        ),
                    )
                except Exception as exc:  # pragma: no cover - tool-specific failures
                    return CallToolObservation(
                        tool_name=action.tool_name,
                        error=ToolError(
                            error_type=ToolErrorType.EXECUTION_ERROR,
                            message=str(exc),
                        ),
                    )

                return CallToolObservation(tool_name=action.tool_name, result=result)

            return self._step_impl(action, timeout_s=timeout_s, **kwargs)

        def _step_impl(
            self,
            action: Action,
            timeout_s: Optional[float] = None,
            **kwargs: Any,
        ) -> Observation:
            raise NotImplementedError

        def close(self) -> None:
            return None

    class EnvClient(Generic[ActT, ObsT, StateT]):
        def __init__(self, base_url: str, timeout_s: float = 30.0):
            self.base_url = base_url.rstrip("/")
            self.timeout_s = timeout_s
            self._client: httpx.AsyncClient | None = None

        async def connect(self) -> "EnvClient[ActT, ObsT, StateT]":
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout_s,
                )
            return self

        async def close(self) -> None:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

        async def __aenter__(self) -> "EnvClient[ActT, ObsT, StateT]":
            return await self.connect()

        async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
            await self.close()

        async def reset(self, **kwargs: Any) -> StepResult[ObsT]:
            await self.connect()
            assert self._client is not None
            response = await self._client.post("/reset", json=kwargs)
            response.raise_for_status()
            return self._parse_result(response.json())

        async def step(self, action: ActT, **kwargs: Any) -> StepResult[ObsT]:
            await self.connect()
            assert self._client is not None
            payload = {"action": self._step_payload(action)}
            response = await self._client.post("/step", json=payload)
            response.raise_for_status()
            return self._parse_result(response.json())

        async def state(self) -> StateT:
            await self.connect()
            assert self._client is not None
            response = await self._client.get("/state")
            response.raise_for_status()
            return self._parse_state(response.json())

        def _step_payload(self, action: ActT) -> Dict[str, Any]:
            if hasattr(action, "model_dump"):
                return action.model_dump()
            return dict(action)

        def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ObsT]:
            raise NotImplementedError

        def _parse_state(self, payload: Dict[str, Any]) -> StateT:
            raise NotImplementedError

    def create_fastapi_app(
        env: Callable[[], MCPEnvironment] | MCPEnvironment,
        action_cls: Type[Action],
        observation_cls: Type[Observation],
        max_concurrent_envs: Optional[int] = None,
        concurrency_config: Any | None = None,
    ) -> FastAPI:
        app = FastAPI(title="PR Review Env (compat)")
        env_instance = env() if callable(env) else env

        @app.get("/health")
        def health() -> Dict[str, str]:
            return {"status": "healthy"}

        @app.post("/reset")
        def reset(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
            observation = env_instance.reset(**(payload or {}))
            return observation.model_dump()

        @app.post("/step")
        def step(payload: Dict[str, Any]) -> Dict[str, Any]:
            action = action_cls.model_validate(payload["action"])
            observation = env_instance.step(action)
            return observation.model_dump()

        @app.get("/state")
        def state() -> Dict[str, Any]:
            return env_instance.state.model_dump()

        return app


def ensure_episode_id(episode_id: Optional[str]) -> str:
    return episode_id or str(uuid.uuid4())
