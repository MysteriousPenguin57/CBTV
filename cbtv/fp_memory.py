"""False-positive spatial memory + negative-sample generators."""
import math


# === [DATA-3] FalsePositiveMemory — spatial FP cache for Bayesian refinement (see Flowchart 17) ===
# Rolling-window store of killed-detection coordinates (up to
# max_history_frames frames). `improved_bayesian_refinement_temporal` queries
# this via spatial proximity to suppress recurring FPs across frames, and
# writes back negatives from the current frame's filter kills at loop end.
# Combined with temporal decay, this is the v6 FP-suppression spine.
class FalsePositiveMemory:
    """
    Maintains a history of false positive detections across frames
    for use in Bayesian refinement of subsequent frames.
    """
    def __init__(self, max_history_frames=10, max_negatives_per_frame=50):
        """Initialize FP memory with configurable history depth."""
        self.history = []  # List of (frame_idx, killed_detections)
        self.max_history_frames = max_history_frames
        self.max_negatives_per_frame = max_negatives_per_frame

        # Spatial grid for fast lookup of recurring false positives
        self.spatial_grid = {}  # (grid_x, grid_y) -> count
        self.grid_size = 20  # pixels

    def add_frame_negatives(self, frame_idx, killed_list):
        """Add killed detections from a frame to history"""
        if not killed_list:
            return

        # Keep only top scoring killed (these are the hardest negatives)
        killed_sorted = sorted(killed_list, key=lambda x: x[3], reverse=True)
        kept_killed = killed_sorted[:self.max_negatives_per_frame]

        # Add to history
        self.history.append((frame_idx, kept_killed))

        # Trim old history
        if len(self.history) > self.max_history_frames:
            self.history.pop(0)

        # Update spatial grid (tracks recurring false positives)
        for x, y, r, s in kept_killed:
            grid_x = int(x // self.grid_size)
            grid_y = int(y // self.grid_size)
            key = (grid_x, grid_y)
            self.spatial_grid[key] = self.spatial_grid.get(key, 0) + 1

    def get_smart_negatives(self, current_positives, frame_shape, min_separation=15.0):
        """
        Get smart negative examples for current frame.
        Prioritizes:
        1. Recurring false positives (high count in spatial grid)
        2. Recent false positives from previous frames
        3. Returns empty if no history (fallback to original method)
        """
        H, W = frame_shape
        negatives = []

        # Priority 1: Recurring false positives from recent history
        recurring_negatives = []
        if self.history:
            # Collect all recent negatives
            all_recent = []
            for frame_idx, killed_list in self.history[-5:]:  # Last 5 frames
                all_recent.extend(killed_list)

            # Find ones that appear multiple times in similar locations
            for x, y, r, s in all_recent:
                grid_x = int(x // self.grid_size)
                grid_y = int(y // self.grid_size)

                # Check recurrence count
                count = self.spatial_grid.get((grid_x, grid_y), 0)
                if count >= 2:  # Appeared at least twice
                    # Check separation from current positives
                    min_dist = min([math.hypot(x-px, y-py)
                                   for px, py, _, _ in current_positives],
                                  default=float('inf'))

                    if min_dist > min_separation:
                        recurring_negatives.append((x, y, r, s, count))

            # Sort by recurrence count (more frequent = harder negative)
            recurring_negatives.sort(key=lambda x: x[4], reverse=True)
            negatives = [(x, y, r, s) for x, y, r, s, _ in recurring_negatives[:30]]

        # Priority 2: Recent false positives (temporal consistency)
        if len(negatives) < 50 and self.history:
            for frame_idx, killed_list in reversed(self.history):
                for x, y, r, s in killed_list:
                    min_dist = min([math.hypot(x-px, y-py)
                                   for px, py, _, _ in current_positives],
                                  default=float('inf'))

                    if min_dist > min_separation and (x, y, r, s) not in negatives:
                        negatives.append((x, y, r, s))

                    if len(negatives) >= 50:
                        break

                if len(negatives) >= 50:
                    break

        return negatives

def generate_offcenter_negatives(positives, rad_use, offsets=(0.3, 0.5, 0.7), angles=8):
    """
    Generate negative examples by perturbing positive detection centers.
    Simulates detection errors (off-center detections).
    """
    negatives = []
    angle_step = 2 * math.pi / angles

    for x, y, r, *rest in positives:
        for offset_factor in offsets:
            offset = offset_factor * r
            for i in range(angles):
                angle = i * angle_step
                dx = offset * math.cos(angle)
                dy = offset * math.sin(angle)
                negatives.append((x + dx, y + dy, r))

    return negatives

def generate_between_particle_negatives(positives, rad_use, min_spacing=2.0, max_spacing=4.0):
    """
    Generate negatives at midpoints between well-separated particles.
    Simulates background regions between particles.
    """
    negatives = []

    for i, (x1, y1, r1, *rest1) in enumerate(positives):
        for x2, y2, r2, *rest2 in positives[i+1:]:
            dist = math.hypot(x2 - x1, y2 - y1)
            max_r = max(r1, r2)

            # Only sample between particles that are well-separated
            if min_spacing * max_r < dist < max_spacing * max_r:
                xm = (x1 + x2) / 2
                ym = (y1 + y2) / 2
                rm = (r1 + r2) / 2
                negatives.append((xm, ym, rm))

    return negatives

def generate_wrong_scale_negatives(positives, scale_factors=(0.5, 0.7, 1.3, 1.5)):
    """
    Generate negatives at correct positions but wrong scales.
    Simulates scale mismatch errors.
    """
    negatives = []

    for x, y, r, *rest in positives:
        for scale in scale_factors:
            negatives.append((x, y, r * scale))

    return negatives

def generate_edge_negatives(H, W, rad_use, border_width=None, spacing=3):
    """
    Generate negatives near frame edges where artifacts occur.
    Simulates edge clutter and boundary artifacts.
    """
    if border_width is None:
        border_width = rad_use

    negatives = []
    step = spacing * rad_use

    # Top and bottom edges
    for x in range(rad_use, W - rad_use, int(step)):
        negatives.append((float(x), float(border_width), float(rad_use)))
        negatives.append((float(x), float(H - border_width), float(rad_use)))

    # Left and right edges
    for y in range(rad_use, H - rad_use, int(step)):
        negatives.append((float(border_width), float(y), float(rad_use)))
        negatives.append((float(W - border_width), float(y), float(rad_use)))

    return negatives

