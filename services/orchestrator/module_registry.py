"""
services/orchestrator/module_registry.py
==========================================
Reads and validates module_registry.yaml at orchestrator startup.

Real schema of module_registry.yaml (services/vision/modalities/<key>):
    services:
      router:
        url: "http://router:8001"
        endpoint: "/route"
      vision:
        url: "http://vision:8002"
        modalities:
          us_breast:
            endpoint: "/analyze/us_breast"
            enabled: true        # optional, default True
          xray:
            endpoint: "/analyze/xray"
            enabled: false       # module not yet active -> orchestrator rejects it
      knowledge:
        url: "http://knowledge:8003"
        endpoint: "/map"

Public API:
    load_module_registry(path) -> ModuleRegistry
    ModuleRegistry.vision_url -> str
    ModuleRegistry.router_url -> str
    ModuleRegistry.knowledge_url -> str
    ModuleRegistry.vision_endpoint_for(module_key) -> str   (raises if disabled/unknown)
"""

import os
import yaml
from dataclasses import dataclass, field


class ModuleRegistryError(Exception):
    """Error when the file is missing, the schema is wrong, or a module is not active."""


@dataclass
class VisionModalityEntry:
    endpoint: str
    enabled: bool = True
    description: str = ""


@dataclass
class ModuleRegistry:
    raw: dict
    router_url: str
    vision_url: str
    knowledge_url: str
    router_endpoint: str
    knowledge_endpoint: str
    vision_modalities: dict = field(default_factory=dict)  # module_key -> VisionModalityEntry

    def vision_endpoint_for(self, module_key: str) -> str:
        """
        Returns the real endpoint for module_key (e.g. 'us_breast' -> '/analyze/us_breast').

        Raises:
            ModuleRegistryError: if module_key does not exist in the registry,
                                  or exists but enabled=false.
        """
        entry = self.vision_modalities.get(module_key)
        if entry is None:
            known = ", ".join(self.vision_modalities.keys())
            raise ModuleRegistryError(
                f"module_key '{module_key}' does not exist in module_registry.yaml. "
                f"Available modules: {known}"
            )
        if not entry.enabled:
            raise ModuleRegistryError(
                f"module_key '{module_key}' is marked enabled: false in "
                "module_registry.yaml - this module is not yet active (roadmap), "
                "the orchestrator must not call it."
            )
        return entry.endpoint


def _env_override(value: str, env_key: str) -> str:
    """Allows overriding the URL via an env var; the endpoint path always comes from YAML."""

    return os.getenv(env_key, value)


def load_module_registry(path: str = "module_registry.yaml") -> ModuleRegistry:
    """
    Load + validate module_registry.yaml.

    Raises:
        ModuleRegistryError: if the file does not exist, the YAML can't be parsed,
                              or a required section (router/vision/knowledge) is missing.
    """
    if not os.path.exists(path):
        raise ModuleRegistryError(
            f"module_registry.yaml not found at '{path}'. "
            "This file is required - the orchestrator uses it to know which service to route to."
        )

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ModuleRegistryError(f"Failed to parse module_registry.yaml: {e}")

    if not isinstance(data, dict) or "services" not in data:
        raise ModuleRegistryError(
            "module_registry.yaml is missing the top-level key 'services'."
        )

    services = data["services"]
    for required in ("router", "vision", "knowledge"):
        if required not in services:
            raise ModuleRegistryError(
                f"module_registry.yaml is missing section 'services.{required}'."
            )

    router_cfg = services["router"]
    vision_cfg = services["vision"]
    knowledge_cfg = services["knowledge"]

    modalities_raw = vision_cfg.get("modalities", {})
    if not modalities_raw:
        raise ModuleRegistryError(
            "module_registry.yaml: 'services.vision.modalities' is empty - "
            "no module available to route to."
        )

    vision_modalities = {}
    for key, entry in modalities_raw.items():
        if "endpoint" not in entry:
            raise ModuleRegistryError(
                f"module_registry.yaml: modality '{key}' is missing field 'endpoint'."
            )
        vision_modalities[key] = VisionModalityEntry(
            endpoint=entry["endpoint"],
            enabled=entry.get("enabled", True),
            description=entry.get("description", ""),
        )

    return ModuleRegistry(
        raw=data,
        router_url=_env_override(router_cfg.get("url", ""), "ROUTER_URL"),
        vision_url=_env_override(vision_cfg.get("url", ""), "VISION_URL"),
        knowledge_url=_env_override(knowledge_cfg.get("url", ""), "KNOWLEDGE_URL"),
        router_endpoint=router_cfg.get("endpoint", "/route"),
        knowledge_endpoint=knowledge_cfg.get("endpoint", "/map"),
        vision_modalities=vision_modalities,
    )
