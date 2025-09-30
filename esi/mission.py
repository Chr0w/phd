import numpy as np
from dataclasses import dataclass
from enum import Enum


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
    def __init__(self, mission_number: int, mission_type: MissionType, waypoint: Waypoint = None, pause_time: float = None):
        """
        Initialize a mission
        
        Args:
            mission_type (MissionType): Type of mission (MOVE_TO_WAYPOINT or PAUSE)
            waypoint (Waypoint, optional): Target waypoint for MOVE_TO_WAYPOINT missions
        """
        self._mission_number = mission_number
        self._status = StatusType.NOT_STARTED
        self._type = mission_type
        self._waypoint = waypoint
        self._pause_time = pause_time
    
    def __str__(self):
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
    
    def set_status(self, status: StatusType):
        """Set the mission status"""
        self._status = status

    def get_type(self):
        return self._type

    def get_status(self):
        return self._status

    def get_type(self):
        return self._type

    def get_waypoint(self):
        return self._waypoint

    def get_pause_time(self):
        return self._pause_time