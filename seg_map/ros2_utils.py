import typing as _t
import time

# Lazy imports to avoid hard dependency when ROS2 not available
try:
	import rclpy  # type: ignore
	from rclpy.node import Node  # type: ignore
	from std_msgs.msg import Float32  # type: ignore
	from geometry_msgs.msg import Twist  # type: ignore
	_ROS2_AVAILABLE = True
except Exception:
	_ROS2_AVAILABLE = False

def is_available() -> bool:
	return _ROS2_AVAILABLE

class StorageUtilizationPublisher:
	"""
	Small wrapper around a ROS2 publisher for the /storage_utilization topic.
	Safe to construct even if ROS2 is unavailable; methods will no-op.
	"""
	def __init__(self) -> None:
		self._node: _t.Optional["Node"] = None
		self._publisher = None
		self._ok = False
		if not _ROS2_AVAILABLE:
			return
		try:
			if not rclpy.ok():  # type: ignore[name-defined]
				rclpy.init()
			self._node = Node('seg_map_storage_utilization_publisher')  # type: ignore[name-defined]
			self._publisher = self._node.create_publisher(Float32, '/storage_utilization', 10)  # type: ignore[name-defined]
			self._ok = True
		except Exception:
			self._node = None
			self._publisher = None
			self._ok = False

	@property
	def ok(self) -> bool:
		return self._ok and self._publisher is not None

	def publish_ratio(self, ratio: float) -> None:
		"""Publish a floating-point ratio if ROS2 is available."""
		if not self.ok:
			return
		try:
			msg = Float32()  # type: ignore[name-defined]
			msg.data = float(ratio)
			self._publisher.publish(msg)  # type: ignore[union-attr]
			# Spin the node once to actually send the message
			rclpy.spin_once(self._node, timeout_sec=0.0)  # type: ignore[name-defined]
		except Exception:
			# Swallow exceptions to avoid crashing the sim due to ROS issues
			pass

	def shutdown(self) -> None:
		if not self._node:
			return
		try:
			self._node.destroy_node()
		except Exception:
			pass


class TeleopCommandSubscriber:
	"""
	Listens to a Twist topic (default /cmd_vel) and stores the latest command.
	Safe to construct when ROS2 is unavailable; methods will no-op.
	"""
	def __init__(self, topic_name: str = '/cmd_vel') -> None:
		self._node: _t.Optional["Node"] = None
		self._subscription = None
		self._ok = False
		self._topic_name = topic_name
		self._last_linear_x = 0.0
		self._last_linear_y = 0.0
		self._last_angular_z = 0.0
		self._last_msg_time = 0.0
		if not _ROS2_AVAILABLE:
			return
		try:
			if not rclpy.ok():  # type: ignore[name-defined]
				rclpy.init()
			self._node = Node('seg_map_teleop_subscriber')  # type: ignore[name-defined]
			self._subscription = self._node.create_subscription(  # type: ignore[name-defined]
				Twist,
				self._topic_name,
				self._on_twist,
				10,
			)
			self._ok = True
		except Exception:
			self._node = None
			self._subscription = None
			self._ok = False

	@property
	def ok(self) -> bool:
		return self._ok and self._node is not None and self._subscription is not None

	def _on_twist(self, msg: "Twist") -> None:
		try:
			self._last_linear_x = float(msg.linear.x)
			self._last_linear_y = float(msg.linear.y)
			self._last_angular_z = float(msg.angular.z)
			self._last_msg_time = time.monotonic()
		except Exception:
			pass

	def spin_once(self) -> None:
		if not self.ok:
			return
		try:
			rclpy.spin_once(self._node, timeout_sec=0.0)  # type: ignore[name-defined]
		except Exception:
			pass

	def get_latest_command(self, stale_timeout_sec: float = 0.5) -> _t.Tuple[float, float, float]:
		"""
		Returns (linear_x, linear_y, angular_z). If command is stale, returns zeros.
		"""
		if not self.ok:
			return (0.0, 0.0, 0.0)
		if self._last_msg_time <= 0.0:
			return (0.0, 0.0, 0.0)
		if (time.monotonic() - self._last_msg_time) > stale_timeout_sec:
			return (0.0, 0.0, 0.0)
		return (self._last_linear_x, self._last_linear_y, self._last_angular_z)

	def shutdown(self) -> None:
		if not self._node:
			return
		try:
			self._node.destroy_node()
		except Exception:
			pass


def compute_storage_utilization(occupied_bins: int, total_bins: int) -> float:
	if total_bins <= 0:
		return 0.0
	return max(0.0, min(1.0, float(occupied_bins) / float(total_bins)))
