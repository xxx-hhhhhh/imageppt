from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProviderSettingsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = False
    apiKey: str = ""
    baseUrl: str = ""
    model: str = ""
    clearApiKey: bool = False


class VisionSettingsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    selectedProvider: str = Field(default="qwen", pattern="^(qwen|local)$")
    mode: str = Field(default="standard", pattern="^(fast|standard|high)$")
    timeout: int = Field(default=120, ge=10, le=600)
    proxyMode: str = Field(default="system", pattern="^(system|manual|none)$")
    manualProxy: str = ""
    providers: dict[str, ProviderSettingsPayload] = Field(default_factory=dict)


class VisionSettingsResponse(BaseModel):
    enabled: bool
    selectedProvider: str
    providers: dict[str, dict]
    configured: bool
    mode: str
    timeout: int
    proxyMode: str
    manualProxy: str
    network: dict


class VisionTestResponse(BaseModel):
    success: bool
    provider: str
    model: str
    message: str
    diagnostics: dict = {}
