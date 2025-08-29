import numpy as np
import csv
import os


class RobotLogger:
    def __init__(self, log_interval=0.1, stop_logging_time=15.0):
        """
        Initialize robot trajectory logger
        
        Args:
            log_interval (float): Time interval between log entries in seconds (default: 0.1s = 10Hz)
            stop_logging_time (float): Time when logging should stop (default: 15.0s)
        """
        self._csv_data = []
        self._last_log_time = 0.0
        self._log_interval = log_interval
        self._stop_logging_time = stop_logging_time
        self._logging_active = True
    
    def log_robot_pose(self, current_time, position, orientation):
        """
        Log robot pose data at specified frequency
        
        Args:
            current_time (float): Current simulation time
            position (np.array): Robot position [x, y, z]
            orientation (np.array): Robot orientation quaternion [w, x, y, z]
        
        Returns:
            bool: True if logging was stopped, False otherwise
        """
        # CSV logging logic
        if self._logging_active and current_time >= self._last_log_time + self._log_interval:
            # Convert quaternion to yaw angle in degrees
            yaw_degrees = self._quaternion_to_yaw_degrees(orientation)
            
            # Log data at specified frequency
            self._csv_data.append([
                current_time,
                position[0], position[1], position[2],  # x, y, z position
                orientation[0], orientation[1], orientation[2], orientation[3],  # quaternion
                yaw_degrees  # yaw angle in degrees
            ])
            self._last_log_time = current_time
        
        # Stop logging when time reaches the defined value
        if self._logging_active and current_time >= self._stop_logging_time:
            self._logging_active = False
            self._save_csv_data()
            return True
        
        return False
    
    def _quaternion_to_yaw_degrees(self, quaternion):
        """Convert quaternion to yaw angle in degrees around Z-axis"""
        # Extract quaternion components (w, x, y, z)
        w, x, y, z = quaternion
        
        # Calculate yaw angle from quaternion
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        yaw_radians = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # Convert to degrees
        yaw_degrees = np.degrees(yaw_radians)
        
        # Normalize to [0, 360) degrees
        yaw_degrees = yaw_degrees % 360.0
        
        return yaw_degrees
    
    def _save_csv_data(self):
        """Save the logged data to a CSV file"""
        filename = "robot_trajectory.csv"
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['time', 'pos_x', 'pos_y', 'pos_z', 'quat_w', 'quat_x', 'quat_y', 'quat_z', 'yaw_degrees'])
            # Write data
            writer.writerows(self._csv_data)
        
        print(f"Robot trajectory data saved to: {filepath}")
        print(f"Total data points logged: {len(self._csv_data)}")
    
    def get_logging_status(self):
        """Get current logging status"""
        return self._logging_active
    
    def get_data_count(self):
        """Get number of logged data points"""
        return len(self._csv_data)
