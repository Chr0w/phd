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
from isaacsim.examples.base.base_sample_extension_experimental import BaseSampleUITemplate
from isaacsim.examples.interactive.user_examples.seg_map.SegMap import SegMap
from isaacsim.gui.components.ui_utils import btn_builder


class SegMapExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "SegMap"
        self.category = "1. My Scripts"

        self.sample = SegMap()
        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Setup SegMap Calculation",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_hello_world.html",
            "overview": "This Example introduces the user on how to do cool stuff with Isaac Sim through scripting in asynchronous mode.",
            "sample": self.sample,
        }


        self.ui_handle = SegMapExtensionUI(**ui_kwargs)

        # register the example with examples browser
        get_browser_instance().register_example(
            name=self.example_name,
            ui_hook=self.ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        if hasattr(self, "sample"):
            self.sample.physics_cleanup()
        if hasattr(self, "ui_handle"):
            self.ui_handle.on_shutdown()
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)

        return

class SegMapExtensionUI(BaseSampleUITemplate):
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

    def _on_start_missions_event(self):
        print("Start missions button pressed")
        self.sample.start_missions()
        self.task_ui_elements["Start missions"].enabled = False
        return

    def post_reset_button_event(self):
        self.task_ui_elements["Start missions"].enabled = True
        return

    def post_load_button_event(self):
        self.task_ui_elements["Start missions"].enabled = True
        return

    def post_clear_button_event(self):
        self.task_ui_elements["Start missions"].enabled = False
        return

    def build_task_controls_ui(self):
        with ui.VStack(spacing=5):
            start_missions_btn = {
                "label": "Start missions",
                "type": "button",
                "text": "Start missions",
                "tooltip": "Start waypoint mission execution",
                "on_clicked_fn": self._on_start_missions_event,
            }

            self.task_ui_elements["Start missions"] = btn_builder(**start_missions_btn)
            self.task_ui_elements["Start missions"].enabled = False
