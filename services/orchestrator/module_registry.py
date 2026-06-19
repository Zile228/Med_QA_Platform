"""
services/orchestrator/module_registry.py
==========================================
Đọc và validate module_registry.yaml lúc orchestrator startup.

Schema thật của module_registry.yaml (services/vision/modalities/<key>):
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
            enabled: false       # module chưa active -> orchestrator reject
      knowledge:
        url: "http://knowledge:8003"
        endpoint: "/map"

Public API:
    load_module_registry(path) -> ModuleRegistry
    ModuleRegistry.vision_url -> str
    ModuleRegistry.router_url -> str
    ModuleRegistry.knowledge_url -> str
    ModuleRegistry.vision_endpoint_for(module_key) -> str   (raises nếu disabled/unknown)
"""

import os
import yaml
from dataclasses import dataclass, field


class ModuleRegistryError(Exception):
    """Loi khi file thieu, sai schema, hoac module khong active."""


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
        Trả về endpoint thật cho module_key (vd 'us_breast' -> '/analyze/us_breast').

        Raises:
            ModuleRegistryError: nếu module_key không tồn tại trong registry,
                                  hoặc tồn tại nhưng enabled=false.
        """
        entry = self.vision_modalities.get(module_key)
        if entry is None:
            known = ", ".join(self.vision_modalities.keys())
            raise ModuleRegistryError(
                f"module_key '{module_key}' không tồn tại trong module_registry.yaml. "
                f"Các module hiện có: {known}"
            )
        if not entry.enabled:
            raise ModuleRegistryError(
                f"module_key '{module_key}' bị đánh dấu enabled: false trong "
                "module_registry.yaml - module này chưa active (roadmap), "
                "orchestrator không được gọi tới."
            )
        return entry.endpoint


def _env_override(value: str, env_key: str) -> str:
    """Cho phep override URL bang env var; endpoint path luon lay tu YAML."""
    
    return os.getenv(env_key, value)


def load_module_registry(path: str = "module_registry.yaml") -> ModuleRegistry:
    """
    Load + validate module_registry.yaml.

    Raises:
        ModuleRegistryError: nếu file không tồn tại, không parse được YAML,
                              hoặc thiếu section bắt buộc (router/vision/knowledge).
    """
    if not os.path.exists(path):
        raise ModuleRegistryError(
            f"module_registry.yaml không tìm thấy tại '{path}'. "
            "File này bắt buộc - orchestrator dùng nó để biết route tới service nào."
        )

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ModuleRegistryError(f"module_registry.yaml không parse được: {e}")

    if not isinstance(data, dict) or "services" not in data:
        raise ModuleRegistryError(
            "module_registry.yaml thiếu key 'services' ở top level."
        )

    services = data["services"]
    for required in ("router", "vision", "knowledge"):
        if required not in services:
            raise ModuleRegistryError(
                f"module_registry.yaml thiếu section 'services.{required}'."
            )

    router_cfg = services["router"]
    vision_cfg = services["vision"]
    knowledge_cfg = services["knowledge"]

    modalities_raw = vision_cfg.get("modalities", {})
    if not modalities_raw:
        raise ModuleRegistryError(
            "module_registry.yaml: 'services.vision.modalities' rỗng - "
            "không có module nào để route tới."
        )

    vision_modalities = {}
    for key, entry in modalities_raw.items():
        if "endpoint" not in entry:
            raise ModuleRegistryError(
                f"module_registry.yaml: modality '{key}' thiếu field 'endpoint'."
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
