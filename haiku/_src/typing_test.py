# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for haiku._src.typing."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import module
from haiku._src import test_utils
from haiku._src import typing


class CallableModule(module.Module):

  def __call__(self, a):
    return a


class TypingTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_module_protocol(self):
    self.assertNotIsInstance(object(), typing.ModuleProtocol)
    self.assertIsInstance(module.Module(), typing.ModuleProtocol)
    self.assertIsInstance(CallableModule(), typing.ModuleProtocol)

  @test_utils.transform_and_run
  def test_supports_call(self):
    self.assertIsInstance(CallableModule(), typing.SupportsCall)
    self.assertNotIsInstance(module.Module(), typing.SupportsCall)

if __name__ == '__main__':
  absltest.main()
