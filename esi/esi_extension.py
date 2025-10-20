# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os

import omni.ext
import omni.ui as ui
import asyncio
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.user_examples import ESI
from isaacsim.gui.components.ui_utils import btn_builder


class EsiExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "ESI"
        self.category = "1. My Scripts"

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Setup ESI Calculation",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_hello_world.html",
            "overview": "This Example introduces the user on how to do cool stuff with Isaac Sim through scripting in asynchronous mode.",
            "sample": ESI(),
        }


        ui_handle = EsiExtensionUI(**ui_kwargs)

        # register the example with examples browser
        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)

        return

class EsiExtensionUI(BaseSampleUITemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_extra_frames(self):
        extra_stacks = self.get_extra_frames_handle()
        self.task_ui_elements = {}

        with extra_stacks:
            with ui.CollapsableFrame(
                title="Task Control",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                # style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                self.build_task_controls_ui()

    def _on_add_objects_event(self):
        print("Spawn objects button pressed")
        asyncio.ensure_future(self.sample._on_add_objects_event_async())
        # self.task_ui_elements["Add objects"].enabled = False
        return

    def _on_edit_world_event(self):
        print("Edit world button pressed")
        asyncio.ensure_future(self.sample._on_edit_world_event_async())
        return

    def post_reset_button_event(self):
        self.task_ui_elements["Add objects"].enabled = True
        return

    def post_load_button_event(self):
        self.task_ui_elements["Add objects"].enabled = True
        return

    def post_clear_button_event(self):
        # World needs to be loaded before objects can be added
        self.task_ui_elements["Add objects"].enabled = True
        return

    def build_task_controls_ui(self):
        with ui.VStack(spacing=5):

            dict = {
                "label": "Add objects",
                "type": "button",
                "text": "Spawn objects",
                "tooltip": "1m² objects",
                "on_clicked_fn": self._on_add_objects_event,
            }

            self.task_ui_elements["Add objects"] = btn_builder(**dict)
            self.task_ui_elements["Add objects"].enabled = False

            edit_world_btn = {
                "label": "Edit world",
                "type": "button",
                "text": "Edit world",
                "tooltip": "Edit world",
                "on_clicked_fn": self._on_edit_world_event,
            }

            self.task_ui_elements["Edit world"] = btn_builder(**edit_world_btn)
            self.task_ui_elements["Edit world"].enabled = True
