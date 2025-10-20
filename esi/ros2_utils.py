import typing as _t

# Lazy imports to avoid hard dependency when ROS2 not available
try:
	import rclpy  # type: ignore
	from rclpy.node import Node  # type: ignore
	from std_msgs.msg import Float32  # type: ignore
	_ROS2_AVAILABLE = True
except Exception:
	_ROS2_AVAILABLE = False

def is_available() -> bool:
	return _ROS2_AVAILABLE

class MapIntegrityPublisher:
	"""
	Small wrapper around a ROS2 publisher for the /map_integrity_ratio topic.
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
			self._node = Node('esi_map_integrity_publisher')  # type: ignore[name-defined]
			self._publisher = self._node.create_publisher(Float32, '/map_integrity_ratio', 10)  # type: ignore[name-defined]
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


def compute_integrity_ratio(total_boxes: int, untouched_boxes: int) -> float:
	if total_boxes <= 0:
		return 0.0
	return max(0.0, min(1.0, float(untouched_boxes) / float(total_boxes)))


def compute_untouched_boxes(box_name_to_position: dict, moved_box_names: set) -> int:
	"""Helper to compute untouched box count from tracking structures."""
	try:
		return len(set(box_name_to_position.keys()) - set(moved_box_names))
	except Exception:
		return 0
