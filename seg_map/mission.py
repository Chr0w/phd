import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass
class Waypoint:
    x: float
    y: float

class MissionType(Enum):
    MOVE_TO_WAYPOINT = "MoveToWaypoint"
    PAUSE = "Pause"

class StatusType(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"

class Mission:
    def __init__(
        self,
        mission_number: int,
        mission_type: MissionType,
        waypoint: Optional[Waypoint] = None,
        pause_time: Optional[float] = None,
        plan_number: Optional[int] = None,
        waypoint_id: Optional[str] = None,
        plan_start: Optional[str] = None,
        plan_target: Optional[str] = None,
        plan_path: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize a mission
        
        Args:
            mission_number: Unique identifier for the mission
            mission_type: Type of mission (MOVE_TO_WAYPOINT or PAUSE)
            waypoint: Target waypoint for MOVE_TO_WAYPOINT missions
            pause_time: Duration for PAUSE missions
            plan_number: Generated plan index this mission belongs to
            waypoint_id: Layout waypoint id for this mission leg
            plan_start: Start waypoint id for the plan
            plan_target: Target waypoint id for the plan
            plan_path: Full shortest-path node list for the plan
        """
        self._mission_number: int = mission_number
        self._status: StatusType = StatusType.NOT_STARTED
        self._type: MissionType = mission_type
        self._waypoint: Optional[Waypoint] = waypoint
        self._pause_time: Optional[float] = pause_time
        self._plan_number: Optional[int] = plan_number
        self._waypoint_id: Optional[str] = waypoint_id
        self._plan_start: Optional[str] = plan_start
        self._plan_target: Optional[str] = plan_target
        self._plan_path: Optional[list[str]] = plan_path
    def __str__(self) -> str:
        """String representation of the mission"""
        result = "-------------\n"
        result += f"Mission Number: {self._mission_number}\n"
        result += f"Type: {self._type.value}\n"
        result += f"Status: {self._status.value}"

        if self._type == MissionType.PAUSE and self._pause_time is not None:
            result += f"\nPause Time: {self._pause_time}"
        elif self._type == MissionType.MOVE_TO_WAYPOINT and self._waypoint is not None:
            result += f"\nWaypoint: ({self._waypoint.x}, {self._waypoint.y})"
        result += "\n-------------\n"
        return result
    
    def set_status(self, status: StatusType) -> None:
        """Set the mission status"""
        self._status = status

    def get_type(self) -> MissionType:
        """Get the mission type"""
        return self._type

    def get_status(self) -> StatusType:
        """Get the mission status"""
        return self._status

    def get_waypoint(self) -> Optional[Waypoint]:
        """Get the mission waypoint"""
        return self._waypoint

    def get_pause_time(self) -> Optional[float]:
        """Get the mission pause time"""
        return self._pause_time

    def get_plan_number(self) -> Optional[int]:
        return self._plan_number

    def get_waypoint_id(self) -> Optional[str]:
        return self._waypoint_id

    def get_plan_start(self) -> Optional[str]:
        return self._plan_start

    def get_plan_target(self) -> Optional[str]:
        return self._plan_target

    def get_plan_path(self) -> Optional[list[str]]:
        return self._plan_path