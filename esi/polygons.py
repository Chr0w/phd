import numpy as np
from typing import Tuple
from pxr import UsdGeom, Gf
import omni
import omni.isaac.core.utils.prims as prim_utils


class polygon:
    def __init__(self, coordinates, color, name):
        """
        Args:
            coordinates: List of (x, y) tuples defining the polygon vertices
            color: Color array for the polygon
            name: Name identifier for the polygon
        """
        self.coordinates = coordinates  # List of (x, y) tuples
        self.color = color
        self.name = name


def is_point_in_polygon(point: Tuple[float, float], poly: polygon) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: (x, y) tuple
        poly: polygon object with coordinates attribute
    
    Returns:
        bool: True if point is inside polygon, False otherwise
    """
    x, y = point
    n = len(poly.coordinates)
    inside = False
    
    p1x, p1y = poly.coordinates[0]
    for i in range(1, n + 1):
        p2x, p2y = poly.coordinates[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def distance_to_polygon_edge(point: Tuple[float, float], poly: polygon) -> float:
    """
    Calculate the minimum distance from a point to any edge of a polygon.
    
    Args:
        point: (x, y) tuple
        poly: polygon object with coordinates attribute
    
    Returns:
        float: Minimum distance to polygon edge
    """
    px, py = point
    min_distance = float('inf')
    n = len(poly.coordinates)
    
    for i in range(n):
        p1 = poly.coordinates[i]
        p2 = poly.coordinates[(i + 1) % n]
        
        x1, y1 = p1
        x2, y2 = p2
        
        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1
        
        # Vector from p1 to point
        px1 = px - x1
        py1 = py - y1
        
        # Project point onto line segment
        dot = px1 * dx + py1 * dy
        len_sq = dx * dx + dy * dy
        
        if len_sq == 0:
            # Degenerate edge (p1 == p2)
            dist = np.sqrt(px1 * px1 + py1 * py1)
        else:
            # Parameter t: 0 = at p1, 1 = at p2
            t = max(0, min(1, dot / len_sq))
            
            # Closest point on line segment
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            
            # Distance from point to closest point on segment
            dist = np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
        
        min_distance = min(min_distance, dist)
    
    return min_distance


async def add_polygon_at(poly: polygon, stage):
    """
    Add a polygon mesh to the world.
    
    Args:
        poly: polygon instance with coordinates (list of (x, y) tuples), color, and name
        stage: USD stage
    """
    prim_path = f"/World/polygons/polygon_{poly.name}"
    
    # Create the mesh prim
    mesh_prim = prim_utils.create_prim(
        prim_path,
        "Mesh",
        position=np.array([0.0, 0.0, 0.0])  # Position will be handled by vertex coordinates
    )
    
    # Get the mesh and set its geometry
    mesh = UsdGeom.Mesh.Get(stage, prim_path)
    
    # Convert coordinates to 3D vertices (z=0 for flat polygon)
    vertices = np.array([
        [coord[0], coord[1], 0.0] for coord in poly.coordinates
    ], dtype=np.float32)
    
    # Set points (vertices)
    points_attr = mesh.CreatePointsAttr()
    points_attr.Set(vertices)
    
    # Triangulate the polygon (fan triangulation from first vertex)
    num_vertices = len(poly.coordinates)
    if num_vertices < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    
    # Create triangles: (0, 1, 2), (0, 2, 3), (0, 3, 4), ...
    face_vertex_indices = []
    for i in range(1, num_vertices - 1):
        face_vertex_indices.extend([0, i, i + 1])
    
    face_vertex_indices = np.array(face_vertex_indices, dtype=np.int32)
    face_vertex_counts = np.array([3] * (num_vertices - 2), dtype=np.int32)
    
    face_vertex_indices_attr = mesh.CreateFaceVertexIndicesAttr()
    face_vertex_indices_attr.Set(face_vertex_indices)
    
    face_vertex_counts_attr = mesh.CreateFaceVertexCountsAttr()
    face_vertex_counts_attr.Set(face_vertex_counts)
    
    # Set color
    color_attr = mesh.CreateDisplayColorAttr()
    color_attr.Set([tuple(poly.color)])


def is_point_in_occupiable_space(point: Tuple[float, float], occupiable_space_polygons: list) -> bool:
    """
    Check if a point is inside any of the occupiable space polygons.
    
    Args:
        point: (x, y) tuple
        occupiable_space_polygons: list of polygon objects
    
    Returns:
        bool: True if point is inside any occupiable space polygon, False otherwise
    """
    for poly in occupiable_space_polygons:
        if is_point_in_polygon(point, poly):
            return True
    return False


def is_box_fit_in_occupiable_space(point: Tuple[float, float], box_circumscribed_radius: float, occupiable_space_polygons: list) -> bool:
    """
    Check if a box (with given circumscribed radius) fits within any occupiable space polygon.
    The box fits if the distance from the center to the nearest polygon edge is >= the circumscribed radius.
    
    Args:
        point: (x, y) tuple - center of the box
        box_circumscribed_radius: float - distance from center to corner of the box
        occupiable_space_polygons: list of polygon objects
    
    Returns:
        bool: True if box fits within any occupiable space polygon, False otherwise
    """
    for poly in occupiable_space_polygons:
        # First check if center is inside the polygon
        if is_point_in_polygon(point, poly):
            # Then check if distance to edge is sufficient
            dist_to_edge = distance_to_polygon_edge(point, poly)
            if dist_to_edge >= box_circumscribed_radius:
                return True
    return False

