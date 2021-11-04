# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Haiku types."""

import abc
import typing
from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp
from typing_extensions import Protocol, runtime_checkable  # pylint: disable=multiple-statements,g-multiple-import

# pytype: disable=module-attr
try:
  # Using PyType's experimental support for forward references.
  Module = typing._ForwardRef("haiku.Module")  # pylint: disable=protected-access
except AttributeError:
  Module = Any
# pytype: enable=module-attr

Initializer = Callable[[Sequence[int], Any], jnp.ndarray]
Params = Mapping[str, Mapping[str, jnp.ndarray]]
State = Mapping[str, Mapping[str, jnp.ndarray]]

# Missing JAX types.
PRNGKey = jnp.ndarray  # pylint: disable=invalid-name


@runtime_checkable
class ModuleProtocol(Protocol):
  """Protocol for Module like types."""

  @abc.abstractproperty
  def name(self) -> str:
    pass

  @abc.abstractproperty
  def module_name(self) -> str:
    pass

  @abc.abstractmethod
  def params_dict(self) -> Mapping[str, jnp.array]:
    pass

  @abc.abstractmethod
  def state_dict(self) -> Mapping[str, jnp.array]:
    pass


@runtime_checkable
class SupportsCall(ModuleProtocol, Callable[..., Any], Protocol):
  """Protocol for Module like types that are Callable.

  Being a protocol means you don't need to explicitly extend this type in order
  to support instance checks with it. For example, :class:`Linear` only extends
  :class:`Module`, however since it conforms (e.g. implements ``__call__``) to
  this protocol you can instance check using it::

  >>> assert isinstance(hk.Linear(1), hk.SupportsCall)
  """

  pass
