# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os

import carb.eventdispatcher
import omni.ext
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd
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
        self.ui_handle.set_sample(self.sample)

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
        self._sample_ref = kwargs.get("sample")

    def set_sample(self, sample) -> None:
        self._sample_ref = sample
        if sample is not None:
            sample.set_layout_prewarm_listener(self._on_layout_prewarm_complete)

    @property
    def sample(self):
        if self._sample_ref is not None:
            return self._sample_ref
        return super().sample

    def _set_start_missions_enabled(self, enabled: bool) -> None:
        start_missions = self.task_ui_elements.get("Start missions")
        if start_missions is not None:
            start_missions.enabled = enabled

    def _on_layout_prewarm_complete(self) -> None:
        if not getattr(self._sample_ref, "_layout_instancers_prewarmed", False):
            return
        self._set_start_missions_enabled(True)

    def _revoke_event_subscriptions(self) -> None:
        if self._stage_event_subscription is not None:
            self._stage_event_subscription.reset()
            self._stage_event_subscription = None
        if self._timeline_event_subscription is not None:
            self._timeline_event_subscription.reset()
            self._timeline_event_subscription = None

    def on_shutdown(self) -> None:
        self._revoke_event_subscriptions()
        super().on_shutdown()

    def _safe_on_stage_event(self, event) -> None:
        if self._sample is None:
            return
        self._sample._physics_cleanup()
        load_world = self._buttons.get("Load World") if self._buttons else None
        if load_world is None:
            return
        self._enable_all_buttons(False)
        load_world.enabled = True

    def _safe_reset_on_stop_event(self, event) -> None:
        load_world = self._buttons.get("Load World") if self._buttons else None
        reset = self._buttons.get("Reset") if self._buttons else None
        if load_world is None or reset is None:
            return
        load_world.enabled = False
        reset.enabled = True
        self.post_clear_button_event()

    def _on_load_world(self) -> None:
        async def _on_load_world_async() -> None:
            self._revoke_event_subscriptions()
            await self._sample.load_world_async()
            await omni.kit.app.get_app().next_update_async()

            usd_context = omni.usd.get_context()
            self._stage_event_subscription = carb.eventdispatcher.get_eventdispatcher().observe_event(
                event_name=usd_context.stage_event_name(omni.usd.StageEventType.CLOSED),
                on_event=self._safe_on_stage_event,
                observer_name="seg_map_extension.on_stage_closed",
            )
            self._timeline_event_subscription = carb.eventdispatcher.get_eventdispatcher().observe_event(
                event_name=omni.timeline.GLOBAL_EVENT_STOP,
                on_event=self._safe_reset_on_stop_event,
                observer_name="seg_map_extension._reset_on_stop_event",
            )

            self._enable_all_buttons(True)
            load_world = self._buttons.get("Load World")
            if load_world is not None:
                load_world.enabled = False
            self.post_load_button_event()

        asyncio.ensure_future(_on_load_world_async())

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

    def _on_spawn_all_objects_event(self):
        print("Spawn all objects button pressed")
        asyncio.ensure_future(self.sample._on_spawn_all_objects_event_async())
        return

    def _on_clear_all_objects_event(self):
        print("Clear all objects button pressed")
        asyncio.ensure_future(self.sample._on_clear_all_objects_event_async())
        return

    def _on_start_missions_event(self):
        print("Start missions button pressed")
        self.sample.start_missions()
        self.task_ui_elements["Start missions"].enabled = False
        return

    def post_reset_button_event(self):
        self.task_ui_elements["Spawn all objects"].enabled = True
        self.task_ui_elements["Clear all objects"].enabled = True
        self._set_start_missions_enabled(False)
        return

    def post_load_button_event(self):
        self.task_ui_elements["Spawn all objects"].enabled = True
        self.task_ui_elements["Clear all objects"].enabled = True
        self._set_start_missions_enabled(False)
        return

    def post_clear_button_event(self):
        self.task_ui_elements["Spawn all objects"].enabled = True
        self.task_ui_elements["Clear all objects"].enabled = True
        self._set_start_missions_enabled(
            getattr(self._sample_ref, "_layout_instancers_prewarmed", False)
        )
        return

    def build_task_controls_ui(self):
        with ui.VStack(spacing=5):
            spawn_all_btn = {
                "label": "Spawn all objects",
                "type": "button",
                "text": "Spawn all objects",
                "tooltip": "Clear all objects and spawn random assets in every bin",
                "on_clicked_fn": self._on_spawn_all_objects_event,
            }
            self.task_ui_elements["Spawn all objects"] = btn_builder(**spawn_all_btn)
            self.task_ui_elements["Spawn all objects"].enabled = False

            clear_all_btn = {
                "label": "Clear all objects",
                "type": "button",
                "text": "Clear all objects",
                "tooltip": "Remove all spawned objects from bins",
                "on_clicked_fn": self._on_clear_all_objects_event,
            }
            self.task_ui_elements["Clear all objects"] = btn_builder(**clear_all_btn)
            self.task_ui_elements["Clear all objects"].enabled = False

            start_missions_btn = {
                "label": "Start missions",
                "type": "button",
                "text": "Start missions",
                "tooltip": "Start waypoint mission execution",
                "on_clicked_fn": self._on_start_missions_event,
            }

            self.task_ui_elements["Start missions"] = btn_builder(**start_missions_btn)
            self.task_ui_elements["Start missions"].enabled = False
