from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from core.service_registry import ServiceRegistry
from utils.exceptions import ServiceAlreadyRegisteredError, ServiceNotRegisteredError


class _Greeter(ABC):
    @abstractmethod
    def greet(self) -> str: ...


class _EnglishGreeter(_Greeter):
    def greet(self) -> str:
        return "hello"


class _FrenchGreeter(_Greeter):
    def greet(self) -> str:
        return "bonjour"


def test_register_and_resolve() -> None:
    registry = ServiceRegistry()
    registry.register(_Greeter, _EnglishGreeter())

    resolved = registry.resolve(_Greeter)
    assert resolved.greet() == "hello"


def test_resolve_unregistered_raises() -> None:
    registry = ServiceRegistry()
    with pytest.raises(ServiceNotRegisteredError):
        registry.resolve(_Greeter)


def test_double_register_without_override_raises() -> None:
    registry = ServiceRegistry()
    registry.register(_Greeter, _EnglishGreeter())

    with pytest.raises(ServiceAlreadyRegisteredError):
        registry.register(_Greeter, _FrenchGreeter())


def test_double_register_with_allow_override_replaces() -> None:
    registry = ServiceRegistry()
    registry.register(_Greeter, _EnglishGreeter())
    registry.register(_Greeter, _FrenchGreeter(), allow_override=True)

    assert registry.resolve(_Greeter).greet() == "bonjour"


def test_is_registered() -> None:
    registry = ServiceRegistry()
    assert registry.is_registered(_Greeter) is False

    registry.register(_Greeter, _EnglishGreeter())
    assert registry.is_registered(_Greeter) is True


def test_unregister() -> None:
    registry = ServiceRegistry()
    registry.register(_Greeter, _EnglishGreeter())
    registry.unregister(_Greeter)

    assert registry.is_registered(_Greeter) is False


def test_unregister_unknown_is_noop() -> None:
    registry = ServiceRegistry()
    registry.unregister(_Greeter)  # must not raise
