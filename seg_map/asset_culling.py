from typing import Callable, Optional

import isaac_sim_utils as isu
import robot_utils


class SectionDistanceCuller:
    """
    Deactivate spawned asset sections beyond lidar range.

    Uses section AABB distance so a section is only hidden when every bin in it
    is outside the sensor range. Cheaper than per-instance culling and safe for
    lidar realism.
    """

    def __init__(
        self,
        section_bounds: dict[str, dict[str, float]],
        get_robot_xy: Callable[[], tuple[float, float]],
        get_active_sections: Callable[[], set[str]],
    ) -> None:
        self._section_bounds = section_bounds
        self._get_robot_xy = get_robot_xy
        self._get_active_sections = get_active_sections
        self._section_active: dict[str, bool] = {}
        self._last_update_time = -1.0
        self._enabled = False

    def refresh_bounds(self, section_bounds: dict[str, dict[str, float]]) -> None:
        self._section_bounds = section_bounds

    def reactivate_all(self, stage) -> None:
        for section_id in self._get_active_sections():
            isu.set_section_assets_active(stage, section_id, True)
            self._section_active[section_id] = True

    def update(self, stage, sim_time: float, force: bool = False) -> None:
        active_sections = self._get_active_sections()
        if not active_sections:
            return

        if (
            not force
            and self._last_update_time >= 0.0
            and (sim_time - self._last_update_time) < robot_utils.SECTION_CULL_UPDATE_INTERVAL_S
        ):
            return
        self._last_update_time = sim_time

        robot_x, robot_y = self._get_robot_xy()
        for section_id in active_sections:
            bounds = self._section_bounds.get(section_id)
            if not bounds:
                continue

            distance_m = robot_utils.distance_to_section_bounds(robot_x, robot_y, bounds)
            currently_active = self._section_active.get(section_id, True)
            should_be_active = robot_utils.section_should_be_active(distance_m, currently_active)
            if should_be_active == currently_active and not force:
                continue

            if isu.set_section_assets_active(stage, section_id, should_be_active):
                self._section_active[section_id] = should_be_active

    def on_simulation_playing(self, stage, sim_time: float) -> None:
        self._enabled = True
        self.update(stage, sim_time, force=True)

    def on_simulation_stopped(self, stage) -> None:
        if not self._enabled:
            return
        self.reactivate_all(stage)
        self._enabled = False
        self._last_update_time = -1.0
