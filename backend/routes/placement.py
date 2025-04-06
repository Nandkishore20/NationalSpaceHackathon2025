from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.spatial import KDTree
import heapq
import datetime
from datetime import datetime
import itertools


# Also, import our database helper functions
from database import insert_placement, get_all_placements

router = APIRouter(tags=["placement"])

# ---- Pydantic Models ----

class Coordinates(BaseModel):
    width: float
    depth: float
    height: float


class Position(BaseModel):
    startCoordinates: Coordinates
    endCoordinates: Coordinates


class RearrangementAction(BaseModel):
    step: int
    action: str  # "move", "remove", "place"
    itemId: str
    fromContainer: Optional[str] = None
    fromPosition: Optional[Position] = None
    toContainer: Optional[str] = None
    toPosition: Optional[Position] = None


class Item(BaseModel):
    itemId: str
    name: str
    width: float
    depth: float
    height: float
    priority: int = Field(..., description="1 (lowest) to 100 (highest priority)")
    expiryDate: Optional[str] = None
    usageLimit: Optional[int] = None
    preferredZone: Optional[str] = None


class Container(BaseModel):
    containerId: str
    zone: str
    width: float
    depth: float
    height: float


class PlacementRequest(BaseModel):
    items: List[Item]
    containers: List[Container]


class PlacementResponse(BaseModel):
    success: bool
    placements: List[Dict]
    rearrangements: List[RearrangementAction]

# ---- Pydantic Models.. ----


# ---- Helper Classes ----

class Space:
    __slots__ = ('container_id', 'width_start', 'depth_start', 'height_start', 
                 'width_end', 'depth_end', 'height_end')
    
    def __init__(self, container_id: str, width_start: float, depth_start: float, height_start: float, 
                 width_end: float, depth_end: float, height_end: float):
        self.container_id = container_id
        self.width_start = width_start
        self.depth_start = depth_start
        self.height_start = height_start
        self.width_end = width_end
        self.depth_end = depth_end
        self.height_end = height_end
        
    @property
    def width(self) -> float:
        return self.width_end - self.width_start
        
    @property
    def depth(self) -> float:
        return self.depth_end - self.depth_start
        
    @property
    def height(self) -> float:
        return self.height_end - self.height_start
        
    @property
    def volume(self) -> float:
        return self.width * self.depth * self.height
        
    @property
    def corner_point(self) -> Tuple[float, float, float]:
        """Returns the corner point for KDTree insertion."""
        return (self.width_start, self.depth_start, self.height_start)
    
    def can_fit_item(self, item: Item, allow_rotation: bool = True) -> bool:
        """Check if an item can fit in this space, optionally with rotation."""
        if not allow_rotation:
            return (self.width >= item.width and 
                    self.depth >= item.depth and 
                    self.height >= item.height)
        
        # Try all possible rotations
        dimensions = [item.width, item.depth, item.height]
        space_dims = [self.width, self.depth, self.height]
        
        import itertools
        for perm in itertools.permutations(dimensions):
            if all(space_dims[i] >= perm[i] for i in range(3)):
                return True
                
        return False
    
    def get_best_rotation(self, item: Item) -> Tuple[float, float, float]:
        """
        Get the best rotation of the item for this space to maximize front accessibility.
        In zero gravity, we prefer orientations that minimize depth to improve access.
        """
        item_dims = [item.width, item.depth, item.height]
        space_dims = [self.width, self.depth, self.height]
    
        best_depth = float('inf')
        best_rotation = None
    
        import itertools
        for perm in itertools.permutations(item_dims):
            width, depth, height = perm
            if (space_dims[0] >= width and 
                space_dims[1] >= depth and 
                space_dims[2] >= height):
            
                # Prioritize rotations that result in minimal depth usage
                if depth < best_depth:
                    best_depth = depth
                    best_rotation = perm
    
        return best_rotation
    

class BinPacker:
    def __init__(self, containers: List[Container], items: List[Item]):
        self.items = sorted(items, key=lambda x: (-x.priority, -(x.width * x.depth * x.height)))
        self.containers = containers
        
        # Map containerId to zone for quick lookup
        self.container_zones = {c.containerId: c.zone for c in containers}
        
        # Initialize container states
        self.container_states = {c.containerId: ContainerState(c) for c in containers}
        
        # Load existing placements and build spatial index
        self._load_existing_placements()
        
        # Build spatial indexes for efficient proximity queries
        self.spatial_indexes = self._build_spatial_indexes()
        
        # Group containers by zone for zone preference matching
        self.zone_containers = {}
        for container in containers:
            if container.zone not in self.zone_containers:
                self.zone_containers[container.zone] = []
            self.zone_containers[container.zone].append(container.containerId)
        
        # Cache container volumes for quick access
        self.container_volumes = {c.containerId: c.width * c.depth * c.height for c in containers}
        
        # Store placements and rearrangement actions
        self.placements = []
        self.rearrangements = []
    
    def _load_existing_placements(self):
        """Load existing placements from the database and update container states."""
        from database import get_all_placements
        
        existing_placements = get_all_placements()
        
        # Handle empty result or None
        if not existing_placements:
            return
        
        # Group placements by container for batch processing
        placements_by_container = {}
        for placement in existing_placements:
            container_id = placement["containerId"]
            if container_id not in placements_by_container:
                placements_by_container[container_id] = []
            placements_by_container[container_id].append(placement)
            
        # Update container states in batch
        for container_id, placements in placements_by_container.items():
            if container_id in self.container_states:
                for placement in placements:
                    self.container_states[container_id].add_existing_occupied_space(placement)
    
    def _build_spatial_indexes(self):
        """Build spatial indexes for each container to enable efficient space queries."""
        indexes = {}
        for container_id, state in self.container_states.items():
            if state.free_spaces:
                # Extract corner points and corresponding spaces
                points = [space.corner_point for space in state.free_spaces]
                if points:  # Only build index if there are points
                    indexes[container_id] = (KDTree(points), state.free_spaces)
        return indexes

    def process(self) -> Tuple[List[Dict], List[Dict]]:
        """Process all items and return placements and rearrangements optimized for zero-gravity."""
        # First sort items primarily by priority, not by size
        self.items.sort(key=lambda x: -x.priority)
    
        items_to_rearrange = []
    
        # First pass: Try to place all items starting with highest priority
        for item in self.items:
            placed = False
        
            # Try to place in preferred zone first
            if item.preferredZone and item.preferredZone in self.zone_containers:
                for container_id in self.zone_containers[item.preferredZone]:
                    placed = self._try_place_item(item, container_id)
                    if placed:
                        break
        
            # If not placed in preferred zone, try all containers
            if not placed:
                for container_id in self.container_states:
                    placed = self._try_place_item(item, container_id)
                    if placed:
                        break
        
            # If still not placed, add to rearrangement list
            if not placed:
                items_to_rearrange.append(item)
    
        # Second pass: Try to rearrange items to make space
        if items_to_rearrange:
            self._handle_rearrangements(items_to_rearrange)
    
        return self.placements, self.rearrangements

    def _try_place_item(self, item: Item, container_id: str) -> bool:
        """Try to place an item in a specific container, considering front accessibility."""
        container_state = self.container_states[container_id]
    
        # Get best space considering front accessibility
        best_space, best_rotation = container_state.find_best_space(item, allow_rotation=True)
    
        if best_space and best_rotation:
            # Check if this placement would block front access to high priority items
            if self._would_block_high_priority_access(container_id, best_space, item):
                return False
            
            position = container_state.place_item(item, best_space, best_rotation)
            placement_record = {
                "itemId": item.itemId,
                "containerId": container_id,
                "position": position.dict(),
                "rotated": best_rotation != (item.width, item.depth, item.height)
            }
            self.placements.append(placement_record)
            # Save the placement to the database
            insert_placement(placement_record)
            return True
        return False   
    
    def _would_block_high_priority_access(self, container_id: str, space: Space, new_item: Item) -> bool:
        """
        Check if placing an item in this space would block front access to higher priority items.
        In zero gravity, we only care about items placed behind this one that need access from the front.
        """
        container_state = self.container_states[container_id]
    
        # Calculate the "blocking region" - the area behind the new item from the front perspective
        blocking_width_start = space.width_start
        blocking_width_end = space.width_start + new_item.width
        blocking_height_start = space.height_start
        blocking_height_end = space.height_start + new_item.height
        blocking_depth_start = space.depth_start + new_item.depth  # Everything behind the item
    
        # Check if any higher priority items would be blocked
        for item_id, occupied_space in container_state.occupied_spaces:
            # Find the original item to check its priority
            original_item = next((i for i in self.items if i.itemId == item_id), None)
        
            if not original_item or original_item.priority <= new_item.priority:
                continue  # Skip items with lower or equal priority
            
            # Check if this higher priority item would be blocked
            if (occupied_space.width_start < blocking_width_end and
                occupied_space.width_end > blocking_width_start and
                occupied_space.height_start < blocking_height_end and 
                occupied_space.height_end > blocking_height_start and
                occupied_space.depth_start >= blocking_depth_start):
                return True  # This would block a higher priority item
            
        return False  # No blocking issues found


    def _handle_rearrangements(self, unplaced_items: List[Item]):
        """Handle items that couldn't be placed directly."""
        step_counter = 1
        
        # Try to move lower priority items to make space
        for item in unplaced_items:
            # Sort containers by zone preference
            preferred_containers = []
            if item.preferredZone:
                preferred_containers = [c for c in self.containers if c.zone == item.preferredZone]
            
            other_containers = [c for c in self.containers if c not in preferred_containers]
            candidate_containers = preferred_containers + other_containers
            
            # Find any container where we might move things to make space
            for container in candidate_containers:
                # Calculate if there's enough total volume in the container
                occupied_spaces = self.container_states[container.containerId].occupied_spaces
                
                # Get volumes of all items in this container with lower priority
                lower_priority_items = []
                for occupied_item_id, space in occupied_spaces:
                    # Find the original item to check its priority
                    original_item = next((i for i in self.items if i.itemId == occupied_item_id), None)
                    if original_item and original_item.priority < item.priority:  # Higher number = lower priority
                        lower_priority_items.append((occupied_item_id, space, original_item))
                
                # Sort by priority (lowest priority first to be moved)
                lower_priority_items.sort(key=lambda x: (x[2].priority, -x[1].volume))
                
                # Calculate needed volume
                item_volume = item.width * item.depth * item.height
                
                # Check if moving lower priority items would free enough space
                current_free_volume = sum(space.volume for space in self.container_states[container.containerId].free_spaces)
                
                if current_free_volume >= item_volume:
                    # We have enough free space, but it might be fragmented
                    # Try to consolidate by moving items
                    if self._consolidate_space(container.containerId, item, step_counter):
                        step_counter += 1
                        break
                
                # If we need to remove lower priority items
                volume_to_free = item_volume - current_free_volume
                items_to_move = []
                
                # Select items to move until we have enough volume
                freed_volume = 0
                for item_id, space, original_item in lower_priority_items:
                    items_to_move.append((item_id, space, original_item))
                    freed_volume += space.volume
                    if freed_volume >= volume_to_free:
                        break
                
                if freed_volume >= volume_to_free:
                    # We can free enough space by moving these items
                    for item_id, space, original_item in items_to_move:
                        # Find another container for this item
                        for other_container in self.containers:
                            if other_container.containerId != container.containerId:
                                other_state = self.container_states[other_container.containerId]
                                other_space, other_rotation = other_state.find_best_space(
                                    original_item, allow_rotation=True
                                )
                                
                                if other_space and other_rotation:
                                    # We can move this item here
                                    # Remove from current container
                                    self._remove_item_from_container(
                                        item_id, container.containerId, space, step_counter
                                    )
                                    step_counter += 1
                                    
                                    # Add to new container
                                    new_position = other_state.place_item(original_item, other_space, other_rotation)
                                    
                                    # Record the placement
                                    self.rearrangements.append(
                                        RearrangementAction(
                                            step=step_counter,
                                            action="place",
                                            itemId=item_id,
                                            toContainer=other_container.containerId,
                                            toPosition=new_position
                                        ).dict()
                                    )
                                    step_counter += 1
                                    
                                    # Update the original placement record
                                    for i, placement in enumerate(self.placements):
                                        if placement["itemId"] == item_id:
                                            self.placements[i] = {
                                                "itemId": item_id,
                                                "containerId": other_container.containerId,
                                                "position": new_position.dict(),
                                                "rotated": other_rotation != (
                                                    original_item.width, original_item.depth, original_item.height
                                                )
                                            }
                                            break
                                    
                                    break
                    
                    # Now try to place our item
                    container_state = self.container_states[container.containerId]
                    best_space, best_rotation = container_state.find_best_space(item, allow_rotation=True)
                    
                    if best_space and best_rotation:
                        position = container_state.place_item(item, best_space, best_rotation)
                        
                        # Record the placement action
                        self.rearrangements.append(
                            RearrangementAction(
                                step=step_counter,
                                action="place",
                                itemId=item.itemId,
                                toContainer=container.containerId,
                                toPosition=position
                            ).dict()
                        )
                        step_counter += 1
                        
                        # Record the placement
                        self.placements.append({
                            "itemId": item.itemId,
                            "containerId": container.containerId,
                            "position": position.dict(),
                            "rotated": best_rotation != (item.width, item.depth, item.height)
                        })
                        
                        break
    def _consolidate_space(self, container_id: str, item: Item, step_counter: int) -> bool:
        """Try to consolidate fragmented space by moving items around."""
        container_state = self.container_states[container_id]
        
        # Create a temporary copy of the container state
        temp_state = ContainerState(next(c for c in self.containers if c.containerId == container_id))
        
        # Get all currently placed items
        placed_items = []
        for item_id, space in container_state.occupied_spaces:
            original_item = next((i for i in self.items if i.itemId == item_id), None)
            if original_item:
                placed_items.append((item_id, original_item, space))
        
        # Add our new item to the list and sort by priority and volume
        target_item_tuple = (item.itemId, item, None)
        all_items = [target_item_tuple] + placed_items
        all_items.sort(key=lambda x: (-x[1].priority, -(x[1].width * x[1].depth * x[1].height)))
        
        # Try to place all items in priority order
        success = True
        placements = {}
        
        for item_id, item_obj, _ in all_items:
            best_space, best_rotation = temp_state.find_best_space(item_obj, allow_rotation=True)
            if best_space and best_rotation:
                position = temp_state.place_item(item_obj, best_space, best_rotation)
                placements[item_id] = (position, best_rotation)
            else:
                success = False
                break
        
        if success:
            # We've found a valid rearrangement - now implement it
            
            # First remove all items from the container
            for item_id, _, space in placed_items:
                self._remove_item_from_container(item_id, container_id, space, step_counter)
                step_counter += 1
            
            # Then place all items according to the new arrangement
            for idx, (item_id, item_obj, _) in enumerate(all_items):
                position, rotation = placements[item_id]
                
                # Place the item in the real container
                container_state = self.container_states[container_id]
                best_space, _ = container_state.find_best_space(item_obj, allow_rotation=True)
                if best_space:
                    actual_position = container_state.place_item(item_obj, best_space, rotation)
                    
                    # Record the placement action
                    self.rearrangements.append(
                        RearrangementAction(
                            step=step_counter + idx,
                            action="place",
                            itemId=item_id,
                            toContainer=container_id,
                            toPosition=actual_position
                        ).dict()
                    )
                    
                    # Update or add the placement record
                    if item_id == item.itemId:
                        # This is our target item that was previously unplaced
                        self.placements.append({
                            "itemId": item_id,
                            "containerId": container_id,
                            "position": actual_position.dict(),
                            "rotated": rotation != (item_obj.width, item_obj.depth, item_obj.height)
                        })
                    else:
                        # Update existing placement
                        for i, placement in enumerate(self.placements):
                            if placement["itemId"] == item_id:
                                self.placements[i] = {
                                    "itemId": item_id,
                                    "containerId": container_id,
                                    "position": actual_position.dict(),
                                    "rotated": rotation != (item_obj.width, item_obj.depth, item_obj.height)
                                }
                                break
            
            return True
        
        return False
    

    def _optimize_container_for_zero_gravity(self, container_id: str) -> bool:
        """
        Optimize a container's space utilization for zero-gravity by reorganizing items
        to maximize front accessibility for high priority items.
    
        Returns:
            bool: True if optimization was successful
        """
        container_state = self.container_states[container_id]
    
        # Get all items currently in the container
        current_items = []
        for item_id, space in container_state.occupied_spaces:
            original_item = next((i for i in self.items if i.itemId == item_id), None)
            if original_item:
                current_items.append((item_id, original_item, space))
    
        # Sort by priority (highest first)
        current_items.sort(key=lambda x: -x[1].priority)
    
        # Create a temporary copy of the container
        temp_state = ContainerState(next(c for c in self.containers if c.containerId == container_id))
    
        # Try to place all items with high priority items closer to front
        success = True
        placements = {}
    
        for item_id, item_obj, _ in current_items:
            best_space, best_rotation = temp_state.find_best_space(item_obj, allow_rotation=True)
            if best_space and best_rotation:
                position = temp_state.place_item(item_obj, best_space, best_rotation)
                placements[item_id] = (position, best_rotation)
            else:
                success = False
                break
    
        if success:
            # Implementation of re-arranging the container would go here
            # Similar to the _consolidate_space method but optimized for front access
            return True
    
        return False

    def handle_new_stock_arrival(self, new_items: List[Item]) -> Tuple[List[Dict], List[Dict]]:
        """
        Handle new stock arrivals with intelligent placement suggestions based on
        priority, available space, and existing item arrangements.
    
        Returns:
            Tuple[List[Dict], List[Dict]]: (placements, rearrangement_actions)
        """
        # Sort new items by priority (highest first)
        sorted_items = sorted(new_items, key=lambda x: (-x.priority, -(x.width * x.depth * x.height)))
    
        placements = []
        rearrangements = []
        items_requiring_space = []
    
        # First pass: Try direct placement in preferred zones
        for item in sorted_items:
            placed = False
        
            # Try preferred zone first for high priority items
            if item.preferredZone and item.preferredZone in self.zone_containers:
                # Sort containers within zone by available volume (most free space first)
                preferred_containers = self.zone_containers[item.preferredZone]
                sorted_containers = sorted(
                    preferred_containers,
                    key=lambda c_id: sum(space.volume for space in self.container_states[c_id].free_spaces),
                    reverse=True
                )
            
                for container_id in sorted_containers:
                    if self._try_place_item(item, container_id):
                        placed = True
                        placements.append({
                            "itemId": item.itemId,
                            "containerId": container_id,
                            "position": self.container_states[container_id].get_item_position(item.itemId).dict(),
                            "placement_type": "direct"
                        })
                        break
        
            # If high priority and not placed, try other zones with forward positioning
            if not placed and item.priority >= 70:  # High priority threshold
                forward_containers = self._get_forward_accessible_containers()
                for container_id in forward_containers:
                    if self._try_place_item(item, container_id):
                        placed = True
                        placements.append({
                            "itemId": item.itemId,
                            "containerId": container_id,
                            "position": self.container_states[container_id].get_item_position(item.itemId).dict(),
                            "placement_type": "forward_position"
                        })
                        break
        
            # Try any container if still not placed
            if not placed:
                for container_id in self.container_states:
                    if self._try_place_item(item, container_id):
                        placed = True
                        placements.append({
                            "itemId": item.itemId,
                            "containerId": container_id,
                            "position": self.container_states[container_id].get_item_position(item.itemId).dict(),
                            "placement_type": "any_available"
                        })
                        break
                    
            # If still not placed, add to list requiring rearrangement
            if not placed:
                items_requiring_space.append(item)
    
        # Handle items that need rearrangement
        if items_requiring_space:
            # Group by zones to optimize rearrangements
            items_by_preferred_zone = {}
            for item in items_requiring_space:
                zone = item.preferredZone if item.preferredZone else "default"
                if zone not in items_by_preferred_zone:
                    items_by_preferred_zone[zone] = []
                items_by_preferred_zone[zone].append(item)
        
            # Process each zone group
            for zone, zone_items in items_by_preferred_zone.items():
                # Sort containers by utilization (least utilized first)
                target_containers = self.zone_containers.get(zone, list(self.container_states.keys()))
                sorted_containers = sorted(
                    target_containers,
                    key=lambda c_id: self._calculate_container_utilization(c_id)
                )
            
                for container_id in sorted_containers:
                    # Try smart rearrangement to fit more items
                    new_placements, new_rearrangements = self._optimize_container_for_new_items(
                        container_id, zone_items
                    )
                
                    # Update tracking
                    placements.extend(new_placements)
                    rearrangements.extend(new_rearrangements)
                
                    # Remove successfully placed items
                    placed_ids = {p["itemId"] for p in new_placements}
                    zone_items = [item for item in zone_items if item.itemId not in placed_ids]
                
                    # If all items placed, break
                    if not zone_items:
                        break
            
                # Update the main list of items requiring space
                items_requiring_space = [
                    item for item in items_requiring_space 
                    if item.itemId not in {p["itemId"] for p in placements}
                ]
    
        # Generate recommendations for any remaining unplaced items
        recommendations = self._generate_placement_recommendations(items_requiring_space)
    
        return placements, rearrangements, recommendations


    def _optimize_container_for_new_items(self, container_id: str, items: List[Item]) -> Tuple[List[Dict], List[Dict]]:
        """
        Optimize a container by rearranging lower priority items to make space for new items.
    
        Args:
            container_id: The container to optimize
            items: New items to place
        
        Returns:
            Tuple of new placements and rearrangement actions
        """
        container_state = self.container_states[container_id]
        step_counter = 1
        placements = []
        rearrangements = []
    
        # Get current items in the container
        current_items = []
        for item_id, space in container_state.occupied_spaces:
            # Find original item
            original_item = next((i for i in self.items if i.itemId == item_id), None)
            if original_item:
                current_items.append((item_id, original_item, space))
    
        # Sort current items by priority (lowest first to be candidates for moving)
        current_items.sort(key=lambda x: (x[1].priority, -x[2].volume))
    
        # Calculate required space for new items
        total_new_volume = sum(item.width * item.depth * item.height for item in items)
        current_free_volume = sum(space.volume for space in container_state.free_spaces)
    
        # Check if we need to make more space
        if current_free_volume < total_new_volume:
            volume_to_free = total_new_volume - current_free_volume
            candidates_to_move = []
        
            # Find lower priority items to move
            freed_volume = 0
            for item_id, item, space in current_items:
                # Only move items with lower priority than new items
                if item.priority < min(new_item.priority for new_item in items):
                    candidates_to_move.append((item_id, item, space))
                    freed_volume += space.volume
                    if freed_volume >= volume_to_free:
                        break
        
            # Try to move these items to other containers
            if candidates_to_move and freed_volume > 0:
                for item_id, item, space in candidates_to_move:
                    # Find best alternative container
                    best_alt_container = self._find_best_alternative_container(item, container_id)
                
                    if best_alt_container:
                        # Remove from current container
                        self._remove_item_from_container(item_id, container_id, space, step_counter)
                        rearrangements.append(
                            RearrangementAction(
                                step=step_counter,
                                action="remove",
                                itemId=item_id,
                                fromContainer=container_id,
                                fromPosition=self._space_to_position(space)
                            ).dict()
                        )
                        step_counter += 1

                        # Place in alternative container
                        alt_state = self.container_states[best_alt_container]
                        best_space, best_rotation = alt_state.find_best_space(item, True)

                        if best_space and best_rotation:
                            new_position = alt_state.place_item(item, best_space, best_rotation)
                            rearrangements.append(
                                RearrangementAction(
                                    step=step_counter,
                                    action="place",
                                    itemId=item_id,
                                    toContainer=best_alt_container,
                                    toPosition=new_position
                                ).dict()
                            )
                            step_counter += 1
                        
                            # Update the placement record
                            for i, placement in enumerate(self.placements):
                                if placement["itemId"] == item_id:
                                    self.placements[i] = {
                                        "itemId": item_id,
                                        "containerId": best_alt_container,
                                        "position": new_position.dict(),
                                        "rotated": best_rotation != (item.width, item.depth, item.height)
                                    }
                                    break
    
        # Now try to place all new items
        for item in sorted(items, key=lambda x: -x.priority):
            best_space, best_rotation = container_state.find_best_space(item, True)
            if best_space and best_rotation:
                position = container_state.place_item(item, best_space, best_rotation)
            
                # Record the placement
                placements.append({
                    "itemId": item.itemId,
                    "containerId": container_id,
                    "position": position.dict(),
                    "rotated": best_rotation != (item.width, item.depth, item.height)
                })
            
                # Add placement action
                rearrangements.append(
                    RearrangementAction(
                        step=step_counter,
                        action="place",
                        itemId=item.itemId,
                        toContainer=container_id,
                        toPosition=position
                    ).dict()
                )
                step_counter += 1
    
        return placements, rearrangements
    

    def _remove_item_from_container(self, item_id: str, container_id: str, space: Space, step: int):
        """Remove an item from a container and record the action."""
        # Remove from occupied spaces
        container_state = self.container_states[container_id]
        container_state.occupied_spaces = [(id, s) for id, s in container_state.occupied_spaces if id != item_id]
        
        # Add space back to free spaces
        container_state.free_spaces.append(space)
        container_state._merge_adjacent_spaces()
        
        # Record the removal action
        position = Position(
            startCoordinates=Coordinates(
                width=space.width_start,
                depth=space.depth_start,
                height=space.height_start
            ),
            endCoordinates=Coordinates(
                width=space.width_end,
                depth=space.depth_end,
                height=space.height_end
            )
        )
        
        self.rearrangements.append(
            RearrangementAction(
                step=step,
                action="remove",
                itemId=item_id,
                fromContainer=container_id,
                fromPosition=position
            ).dict()
        )



class ContainerState:
    """Maintains the state of a container with occupied and free spaces."""
    
    def __init__(self, container: Container):
        self.container = container
        self.occupied_spaces = []  
        self.free_spaces = [
            Space(
                container.containerId,
                0, 0, 0,
                container.width, container.depth, container.height
            )
        ]
    
    def add_existing_occupied_space(self, placement: Dict):
        """
        When there is an existing placement (from the DB), remove its volume from free_spaces
        and record it in occupied_spaces.
        """
        pos = placement["position"]
        occupied = Space(
            self.container.containerId,
            pos["startCoordinates"]["width"],
            pos["startCoordinates"]["depth"],
            pos["startCoordinates"]["height"],
            pos["endCoordinates"]["width"],
            pos["endCoordinates"]["depth"],
            pos["endCoordinates"]["height"]
        )
        self.occupied_spaces.append((placement["itemId"], occupied))
        # Remove the occupied space from free_spaces.
        # For simplicity, we assume that if a placement exists, its region is simply removed.
        # (A more complete solution would subtract the volume from free_spaces.)
        for free_space in self.free_spaces.copy():
            if (free_space.width_start <= occupied.width_start and
                free_space.depth_start <= occupied.depth_start and
                free_space.height_start <= occupied.height_start and
                free_space.width_end >= occupied.width_end and
                free_space.depth_end >= occupied.depth_end and
                free_space.height_end >= occupied.height_end):
                self.free_spaces.remove(free_space)
                # Add back the remaining free spaces from splitting the free_space around occupied.
                # Here we add simple splits (right, back, above) like in the original logic.
                if free_space.width_end > occupied.width_end:
                    self.free_spaces.append(Space(
                        self.container.containerId,
                        occupied.width_end, free_space.depth_start, free_space.height_start,
                        free_space.width_end, free_space.depth_end, free_space.height_end
                    ))
                if free_space.depth_end > occupied.depth_end:
                    self.free_spaces.append(Space(
                        self.container.containerId,
                        free_space.width_start, occupied.depth_end, free_space.height_start,
                        occupied.width_end, free_space.depth_end, free_space.height_end
                    ))
                if free_space.height_end > occupied.height_end:
                    self.free_spaces.append(Space(
                        self.container.containerId,
                        free_space.width_start, free_space.depth_start, occupied.height_end,
                        occupied.width_end, occupied.depth_end, free_space.height_end
                    ))
        # You can also call self._merge_adjacent_spaces() here if needed.

    def _get_forward_accessible_containers(self) -> List[str]:
        """
        Identify containers that are easily accessible from the front.
        This would typically be determined by zone characteristics or container positioning.
    
        Returns:
            List of container IDs that are forward accessible
        """
        # In a real system, this might use container metadata or coordinates
        # Here we'll use a simple heuristic based on container utilization
        forward_containers = []
    
        # Consider containers with lower utilization (below 60%) as more accessible
        for container_id in self.container_states:
            utilization = self._calculate_container_utilization(container_id)
            if utilization < 0.6:
                forward_containers.append(container_id)
    
        # Sort by utilization (least utilized first)
        forward_containers.sort(
            key=lambda c_id: self._calculate_container_utilization(c_id)
        )
    
        return forward_containers

    def _calculate_container_utilization(self, container_id: str) -> float:
        """Calculate the current utilization of a container as a ratio of used volume to total volume."""
        container_state = self.container_states[container_id]
        total_volume = self.container_volumes[container_id]
    
        # Calculate used space
        used_volume = 0
        for _, space in container_state.occupied_spaces:
            used_volume += space.volume
    
        return used_volume / total_volume if total_volume > 0 else 0

    def _generate_placement_recommendations(self, unplaced_items: List[Item]) -> List[Dict]:
        """
        Generate recommendations for items that couldn't be placed.
    
        Args:
            unplaced_items: List of items that couldn't be placed
        
        Returns:
            List of recommendation objects
        """
        if not unplaced_items:
            return []
    
        recommendations = []
    
        # Group items by preferred zone
        items_by_zone = {}
        for item in unplaced_items:
            zone = item.preferredZone if item.preferredZone else "any"
            if zone not in items_by_zone:
                items_by_zone[zone] = []
            items_by_zone[zone].append(item)
    
        # Identify containers with low priority items
        low_priority_containers = {}
        for container_id, state in self.container_states.items():
            low_priority_volume = 0
            for item_id, space in state.occupied_spaces:
                # Find the original item
                original_item = next((i for i in self.items if i.itemId == item_id), None)
                if original_item and original_item.priority < 30:  # Threshold for "low priority"
                    low_priority_volume += space.volume
        
            if low_priority_volume > 0:
                low_priority_containers[container_id] = low_priority_volume
    
        # Generate recommendations for each zone
        for zone, zone_items in items_by_zone.items():
            total_volume_needed = sum(item.width * item.depth * item.height for item in zone_items)
        
            # Recommend containers with low priority items that could be moved
            potential_containers = []
            for container_id, low_pri_volume in low_priority_containers.items():
                container_zone = self.container_zones[container_id]
                zone_match = zone == "any" or container_zone == zone
            
                if zone_match and low_pri_volume >= total_volume_needed * 0.7:  # 70% threshold
                    potential_containers.append({
                        "containerId": container_id,
                        "zone": container_zone,
                        "availableVolumeFromLowPriority": low_pri_volume,
                        "currentUtilization": self._calculate_container_utilization(container_id)
                })  
        
            if potential_containers:
                recommendations.append({
                    "zoneRequested": zone,
                    "itemsUnplaced": [{"itemId": item.itemId, "priority": item.priority} for item in zone_items],
                    "volumeNeeded": total_volume_needed,
                    "recommendation": "rearrange_low_priority",
                    "potentialContainers": sorted(potential_containers, 
                                               key=lambda c: c["availableVolumeFromLowPriority"], 
                                               reverse=True)[:3]  # Top 3 options
                })
            else:
                # No good containers with low priority items - recommend consolidation
                recommendations.append({
                    "zoneRequested": zone,
                    "itemsUnplaced": [{"itemId": item.itemId, "priority": item.priority} for item in zone_items],
                    "volumeNeeded": total_volume_needed,
                    "recommendation": "consolidate_containers",
                    "reason": "No containers with sufficient low-priority items found"
                })
    
        return recommendations
        
    def find_best_space(self, item: Item, allow_rotation: bool = True) -> Tuple[Optional[Space], Optional[Tuple[float, float, float]]]:
        """Find the best space to place an item using Best Fit strategy optimized for zero-gravity and front access."""
        item_volume = item.width * item.depth * item.height
    
        candidate_spaces = []
    
        # Evaluate all free spaces
        for space in self.free_spaces:
            if space.can_fit_item(item, allow_rotation):
                # Calculate accessibility score - prefer spaces closer to front (depth=0)
                # Lower depth_start means better front accessibility
                accessibility_score = space.depth_start

                # Calculate space efficiency
                space_efficiency = item_volume / space.volume
            
                candidate_spaces.append((space, accessibility_score, space_efficiency))

        if not candidate_spaces:
            return None, None

        # Sort first by accessibility (lower is better), then by efficiency (higher is better)
        candidate_spaces.sort(key=lambda x: (x[1], -x[2]))

        best_space = candidate_spaces[0][0]    
        if allow_rotation:
            best_rotation = best_space.get_best_rotation(item)
            return best_space, best_rotation
    
        return best_space, (item.width, item.depth, item.height)
    
    def place_item(self, item: Item, space: Space, rotated_dimensions: Tuple[float, float, float]) -> Position:
        """Place an item in a space and update free spaces."""
        width, depth, height = rotated_dimensions
        
        # Create position for the placement
        position = Position(
            startCoordinates=Coordinates(
                width=space.width_start,
                depth=space.depth_start,
                height=space.height_start
            ),
            endCoordinates=Coordinates(
                width=space.width_start + width,
                depth=space.depth_start + depth,
                height=space.height_start + height
            )
        )
        
        # Remove the space that we are using
        self.free_spaces.remove(space)
        
        # Generate new free spaces (up to 3 new spaces from the remaining volume)
        # Space to the right of the item
        if space.width_end > space.width_start + width:
            self.free_spaces.append(Space(
                self.container.containerId,
                space.width_start + width, space.depth_start, space.height_start,
                space.width_end, space.depth_end, space.height_end
            ))
            
        # Space to the back of the item
        if space.depth_end > space.depth_start + depth:
            self.free_spaces.append(Space(
                self.container.containerId,
                space.width_start, space.depth_start + depth, space.height_start,
                space.width_start + width, space.depth_end, space.height_end
            ))
            
        # Space above the item
        if space.height_end > space.height_start + height:
            self.free_spaces.append(Space(
                self.container.containerId,
                space.width_start, space.depth_start, space.height_start + height,
                space.width_start + width, space.depth_start + depth, space.height_end
            ))
        
        # Merge adjacent free spaces when possible to prevent fragmentation
        self._merge_adjacent_spaces()
        
        # Add to occupied spaces
        occupied_space = Space(
            self.container.containerId,
            position.startCoordinates.width, position.startCoordinates.depth, position.startCoordinates.height,
            position.endCoordinates.width, position.endCoordinates.depth, position.endCoordinates.height
        )
        self.occupied_spaces.append((item.itemId, occupied_space))
        
        return position
    
    def _merge_adjacent_spaces(self):
        """Merge adjacent free spaces to reduce fragmentation."""
        # This is a simplified merge - a production system would need a more sophisticated algorithm
        i = 0
        while i < len(self.free_spaces):
            j = i + 1
            while j < len(self.free_spaces):
                # Check if spaces can be merged (simplified check)
                if self._can_merge(self.free_spaces[i], self.free_spaces[j]):
                    self.free_spaces[i] = self._merge_spaces(self.free_spaces[i], self.free_spaces[j])
                    self.free_spaces.pop(j)
                else:
                    j += 1
            i += 1
    
    def _can_merge(self, space1: Space, space2: Space) -> bool:
        """Check if two spaces can be merged (simplified)."""
        # This is a simplified check - production would use a more robust algorithm
        return (
            # Check if spaces are adjacent in width dimension
            (space1.width_end == space2.width_start or space2.width_end == space1.width_start) and
            space1.depth_start == space2.depth_start and
            space1.depth_end == space2.depth_end and
            space1.height_start == space2.height_start and
            space1.height_end == space2.height_end
        ) or (
            # Check if spaces are adjacent in depth dimension
            (space1.depth_end == space2.depth_start or space2.depth_end == space1.depth_start) and
            space1.width_start == space2.width_start and
            space1.width_end == space2.width_end and
            space1.height_start == space2.height_start and
            space1.height_end == space2.height_end
        ) or (
            # Check if spaces are adjacent in height dimension
            (space1.height_end == space2.height_start or space2.height_end == space1.height_start) and
            space1.width_start == space2.width_start and
            space1.width_end == space2.width_end and
            space1.depth_start == space2.depth_start and
            space1.depth_end == space2.depth_end
        )
    
    def _merge_spaces(self, space1: Space, space2: Space) -> Space:
        """Merge two adjacent spaces."""
        return Space(
            self.container.containerId,
            min(space1.width_start, space2.width_start),
            min(space1.depth_start, space2.depth_start),
            min(space1.height_start, space2.height_start),
            max(space1.width_end, space2.width_end),
            max(space1.depth_end, space2.depth_end),
            max(space1.height_end, space2.height_end)
        )




def _load_existing_placements_fixed(packer):
        """Load existing placements from the database and update container states."""
        from database import get_all_placements
    
        existing_placements = get_all_placements()
    
        # Handle empty result or None
        if not existing_placements:
            return
    
        # Group placements by container for batch processing
        placements_by_container = {}
        for placement in existing_placements:
            container_id = placement["containerId"]
            if container_id not in placements_by_container:
                placements_by_container[container_id] = []
            placements_by_container[container_id].append(placement)
        
        # Update container states in batch
        for container_id, placements in placements_by_container.items():
            if container_id in packer.container_states:
                for placement in placements:
                    packer.container_states[container_id].add_existing_occupied_space(placement)
    
        # Build spatial indexes after loading placements
        packer.spatial_indexes = packer._build_spatial_indexes()

# ---- API Endpoints ----

@router.post("/", response_model=PlacementResponse)
async def place_items(data: PlacementRequest):
    """
    Plan optimal cargo placement based on container and item properties.
    This version first loads existing placements from the database.
    """
    try:
        # Import here to ensure we have access to the function
        from database import get_all_placements
        
        # Initialize the bin packer
        packer = BinPacker(data.containers, data.items)
        
        # Fix: The _load_existing_placements method is defined but has an indentation issue
        # We need to properly implement and bind this method to the instance
        packer._load_existing_placements = lambda: _load_existing_placements_fixed(packer)
        
        # Call it explicitly before processing
        packer._load_existing_placements()
        
        # Process the items
        placements, rearrangements = packer.process()
        
        # Prepare response
        response = {
            "success": len(placements) == len(data.items),
            "placements": placements,
            "rearrangements": rearrangements
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing placement: {str(e)}")

@router.post("/new-stock", response_model=PlacementResponse)
async def handle_new_stock(data: PlacementRequest):
    """
    Handle new stock arrivals with intelligent placement suggestions.
    """
    try:
        # Get existing containers from database
        from database import get_all_containers, get_all_items
        
        # Get all existing items and containers
        db_containers = [Container(**container) for container in get_all_containers()]
        existing_items = [Item(**item) for item in get_all_items()]
        
        # Initialize bin packer with existing state
        packer = BinPacker(db_containers, existing_items)
        
        # Process new items
        placements, rearrangements, recommendations = packer.handle_new_stock_arrival(data.items)
        
        return {
            "success": len(placements) == len(data.items),
            "placements": placements,
            "rearrangements": rearrangements,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing new stock: {str(e)}")


@router.post("/simulate")
async def simulate_placement(data: PlacementRequest):
    """
    Simulate placement without making actual changes.
    Useful for testing different placement strategies.
    """
    try:
        # Initialize the bin packer
        packer = BinPacker(data.containers, data.items)
        
        # Process the items
        placements, rearrangements = packer.process()
        
        # Calculate metrics
        total_items = len(data.items)
        placed_items = len(placements)
        placement_rate = placed_items / total_items if total_items > 0 else 0
        
        # Container utilization
        container_volumes = {c.containerId: c.width * c.depth * c.height for c in data.containers}
        used_volumes = {c.containerId: 0 for c in data.containers}
        
        for placement in placements:
            container_id = placement["containerId"]
            pos = placement["position"]
            width = pos["endCoordinates"]["width"] - pos["startCoordinates"]["width"]
            depth = pos["endCoordinates"]["depth"] - pos["startCoordinates"]["depth"]
            height = pos["endCoordinates"]["height"] - pos["startCoordinates"]["height"]
            used_volumes[container_id] += width * depth * height
        
        container_utilization = {
            c_id: used_volumes.get(c_id, 0) / container_volumes.get(c_id, 1) 
            for c_id in container_volumes
        }
        
        # Prepare response with metrics
        response = {
            "success": placed_items == total_items,
            "placements": placements,
            "rearrangements": rearrangements,
            "metrics": {
                "totalItems": total_items,
                "placedItems": placed_items,
                "placementRate": placement_rate,
                "containerUtilization": container_utilization,
                "averageUtilization": sum(container_utilization.values()) / len(container_utilization) 
                                      if container_utilization else 0
            }
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error simulating placement: {str(e)}")



@router.post("/optimize")
async def optimize_placement(data: PlacementRequest):
    """
    Optimize existing cargo placements to maximize space efficiency.
    Useful for periodic maintenance and reorganization.
    """
    try:
        # Get the current state (we'd typically get this from a database)
        # For this endpoint, we'll simulate by doing an initial placement
        packer = BinPacker(data.containers, data.items)
        initial_placements, _ = packer.process()
        
        # Now try to optimize by repacking everything
        # In a real system, we'd read the current state from the database
        
        # Extract items from placements
        current_items = []
        for item in data.items:
            if any(p["itemId"] == item.itemId for p in initial_placements):
                current_items.append(item)
        
        # Sort items by priority, then by volume (largest first)
        current_items.sort(key=lambda x: (-x.priority, (x.width * x.depth * x.height)))
        
        # Create a new packer with the current items
        optimizer = BinPacker(data.containers, current_items)
        optimized_placements, rearrangements = optimizer.process()
        
        # Calculate improvement metrics
        original_volume_usage = {}
        optimized_volume_usage = {}
        
        for container in data.containers:
            container_id = container.containerId
            container_volume = container.width * container.depth * container.height
            
            # Calculate original usage
            original_used = 0
            for p in initial_placements:
                if p["containerId"] == container_id:
                    pos = p["position"]
                    width = pos["endCoordinates"]["width"] - pos["startCoordinates"]["width"]
                    depth = pos["endCoordinates"]["depth"] - pos["startCoordinates"]["depth"]
                    height = pos["endCoordinates"]["height"] - pos["startCoordinates"]["height"]
                    original_used += width * depth * height
            
            # Calculate optimized usage
            optimized_used = 0
            for p in optimized_placements:
                if p["containerId"] == container_id:
                    pos = p["position"]
                    width = pos["endCoordinates"]["width"] - pos["startCoordinates"]["width"]
                    depth = pos["endCoordinates"]["depth"] - pos["startCoordinates"]["depth"]
                    height = pos["endCoordinates"]["height"] - pos["startCoordinates"]["height"]
                    optimized_used += width * depth * height
            
            original_volume_usage[container_id] = original_used / container_volume
            optimized_volume_usage[container_id] = optimized_used / container_volume

            # Calculate overall improvement metrics
            avg_original_usage = sum(original_volume_usage.values()) / len(original_volume_usage) if original_volume_usage else 0
            avg_optimized_usage = sum(optimized_volume_usage.values()) / len(optimized_volume_usage) if optimized_volume_usage else 0
            space_efficiency_improvement = avg_optimized_usage - avg_original_usage

        # Create response
        response = {
            "success": True,
            "original_placements": initial_placements,
            "optimized_placements": optimized_placements,
            "rearrangements": rearrangements,
            "metrics": {
                "original_volume_usage": original_volume_usage,
                "optimized_volume_usage": optimized_volume_usage,
                "average_original_usage": avg_original_usage,
                "average_optimized_usage": avg_optimized_usage,
                "space_efficiency_improvement": space_efficiency_improvement,
                "total_rearrangements": len(rearrangements)
            }
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing placement: {str(e)}")


@router.get("/container/{container_id}/utilization")
async def get_container_utilization(container_id: str):
    """
    Get detailed utilization metrics for a specific container.
    """
    try:
        # In a real system, we would query the database for the container and its contents
        # For this example, we'll return a mock response
        
        return {
            "containerId": container_id,
            "volumeUtilization": 0.75,  # 75% of container volume used
            "itemCount": 12,
            "priorityDistribution": {
                "priority1": 3,
                "priority2": 5,
                "priority3": 2,
                "priority4": 1,
                "priority5": 1
            },
            "expiryItems": [
                {"itemId": "item123", "name": "Medical Supply", "expiryDate": "2024-05-15"}
            ],
            "zoneCompliance": 0.92,  # 92% of items in their preferred zones
            "fragmentation": 0.15,   # 15% space fragmentation
            "recommendations": [
                "Consider consolidating priority 4-5 items to free up space",
                "Move item123 to priority access area due to upcoming expiry"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting container utilization: {str(e)}")






# Add new endpoint
@router.post("/auto-place", response_model=PlacementResponse)
async def auto_place_items():
    """
    Automatically place only new/unplaced items from the database
    while preserving existing placements
    """
    try:
        # Fetch all items and containers from database
        from database import get_all_items, get_all_containers, get_all_placements
        
        # Get all items and containers
        db_items = [Item(**item) for item in get_all_items()]
        db_containers = [Container(**container) for container in get_all_containers()]
        
        # Get existing placements
        existing_placements = get_all_placements()
        existing_item_ids = {placement["itemId"] for placement in existing_placements}
        
        # Filter to only include new/unplaced items
        new_items = [item for item in db_items if item.itemId not in existing_item_ids]
        
        # Create request object with only new items
        placement_request = PlacementRequest(
            items=new_items,
            containers=db_containers
        )
        
        # Process only the new items
        packer = BinPacker(placement_request.containers, placement_request.items)
        
        # Fix: The _load_existing_placements method is defined but has an indentation issue
        # We need to call it manually since it's not being called in the constructor
        packer._load_existing_placements = lambda: packer._load_existing_placements_impl(existing_placements)
        packer._load_existing_placements_impl = lambda placements: _load_existing_placements_fixed(packer)
        
        # Call it explicitly before processing
        packer._load_existing_placements()
        
        # Now process the placements
        new_placements, rearrangements = packer.process()
        
        # Get all placements (including existing ones)
        all_placements = get_all_placements()
        
        return {
            "success": len(new_placements) == len(new_items),
            "placements": all_placements,  # Return all placements, not just the new ones
            "rearrangements": rearrangements
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing placement: {str(e)}")
    
@router.get("/placements", response_model=List[Dict])
async def get_all_placements_endpoint():
    """Retrieve all placement records from the database"""
    try:
        placements = get_all_placements()
        return placements
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@router.post("/new-item", response_model=PlacementResponse)
async def handle_new_stock(data: PlacementRequest):
    """
    Handle new stock arrivals, placing items into the containers provided in the request.
    """
    try:
        from database import get_all_placements, get_all_items

        # Extract container IDs from the request
        request_container_ids = [c.containerId for c in data.containers]

        # Fetch all existing placements from the database
        all_placements = get_all_placements()

        # Filter placements to only those in the requested containers
        relevant_placements = [p for p in all_placements if p["containerId"] in request_container_ids]

        # Collect existing item IDs from relevant placements
        existing_item_ids = {p["itemId"] for p in relevant_placements}

        # Fetch all existing items from the database
        all_existing_items = [Item(**item) for item in get_all_items()]

        # Filter existing items to those present in the relevant placements
        existing_items = [item for item in all_existing_items if item.itemId in existing_item_ids]

        # Combine existing items with new items from the request
        all_items = existing_items + data.items

        # Initialize BinPacker with the requested containers and all items (existing + new)
        packer = BinPacker(data.containers, all_items)

        # Load relevant existing placements into the packer's container states
        def load_relevant_placements():
            placements_by_container = {}
            for placement in relevant_placements:
                container_id = placement["containerId"]
                if container_id not in placements_by_container:
                    placements_by_container[container_id] = []
                placements_by_container[container_id].append(placement)
            
            for container_id, placements in placements_by_container.items():
                if container_id in packer.container_states:
                    for placement in placements:
                        packer.container_states[container_id].add_existing_occupied_space(placement)
            
            # Rebuild spatial indexes after loading placements
            packer.spatial_indexes = packer._build_spatial_indexes()

        load_relevant_placements()

        # Process only the new items through handle_new_stock_arrival
        placements, rearrangements, recommendations = packer.handle_new_stock_arrival(data.items)

        # Filter placements to include only new items
        new_item_ids = {item.itemId for item in data.items}
        new_placements = [p for p in placements if p["itemId"] in new_item_ids]

        return {
            "success": len(new_placements) == len(data.items),
            "placements": new_placements,
            "rearrangements": rearrangements,
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing new stock: {str(e)}")
