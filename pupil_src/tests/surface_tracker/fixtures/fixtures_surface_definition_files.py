"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os
import tempfile
import typing
import uuid

from file_methods import Persistent_Dict

from . import surfaces_serialized_v00_square, surfaces_serialized_v01_mixed

__all__ = [
    "surface_definition_v00_dir",
    "surface_definition_v01_before_update_dir",
    "surface_definition_v01_after_update_dir",
]


def surface_definition_v00_dir() -> str:
    """
    Create directory containing surface definition file generated by the app before version 1.15 release
    """
    return _create_dir_with_surface_definition_file(
        file_name="surface_definitions",
        serialized_surfaces=surfaces_serialized_v00_square(),
    )


def surface_definition_v01_before_update_dir() -> str:
    """
    Create directory containing surface definition file generated by the app at version 1.15 release
    """
    return _create_dir_with_surface_definition_file(
        file_name="surface_definitions",
        serialized_surfaces=surfaces_serialized_v01_mixed(),
    )


def surface_definition_v01_after_update_dir() -> str:
    """
    Create directory containing surface definition file generated by the app after version 1.15 release
    """
    return _create_dir_with_surface_definition_file(
        file_name="surface_definitions_v01",
        serialized_surfaces=surfaces_serialized_v01_mixed(),
    )


##### PRIVATE


def _create_dir_with_surface_definition_file(
    file_name: str, serialized_surfaces: typing.Collection[dict]
) -> str:
    root_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    file_path = os.path.join(root_dir, file_name)

    os.makedirs(root_dir)

    assert not os.path.exists(file_path)
    surfaces_file = Persistent_Dict(file_path)
    surfaces_file["surfaces"] = serialized_surfaces
    surfaces_file.save()
    assert os.path.isfile(file_path)

    return root_dir
