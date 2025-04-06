import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from datetime import datetime, date


from database import get_all_items, get_all_containers, get_all_placements

# Create a router for waste management routes
router = APIRouter()

# Pydantic models for request and response
class Coordinates(BaseModel):
    width: float
    depth: float
    height: float

class Position(BaseModel):
    startCoordinates: Coordinates
    endCoordinates: Coordinates

class WasteItem(BaseModel):
    itemId: str
    name: str
    reason: str
    containerId: str
    position: Position

class WasteIdentifyResponse(BaseModel):
    success: bool
    wasteItems: List[WasteItem]

class ReturnPlanRequest(BaseModel):
    undockingContainerId: str
    undockingDate: str  # ISO format
    maxWeight: float

class ReturnStep(BaseModel):
    step: int
    itemId: str
    itemName: str
    fromContainer: str
    toContainer: str

class RetrievalStep(BaseModel):
    step: int
    action: str  # "remove", "setAside", "retrieve", "placeBack"
    itemId: str
    itemName: str

class ReturnItem(BaseModel):
    itemId: str
    name: str
    reason: str

class ReturnManifest(BaseModel):
    undockingContainerId: str
    undockingDate: str
    returnItems: List[ReturnItem]
    totalVolume: float
    totalWeight: float

# FIX: Added ReturnPlanResponse model that was missing
class ReturnPlanResponse(BaseModel):
    success: bool
    returnPlan: List[ReturnStep]
    retrievalSteps: List[RetrievalStep]
    returnManifest: ReturnManifest

class UndockingRequest(BaseModel):
    undockingContainerId: str
    timestamp: str  # ISO format

class UndockingResponse(BaseModel):
    success: bool
    itemsRemoved: int

def get_db_connection():
    """Create and return a database connection with row factory"""
    conn = sqlite3.connect('inventory.db')
    conn.row_factory = sqlite3.Row
    return conn

def setup_database():
    """Ensure all required tables exist in the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create waste_log table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS waste_log (
        item_id TEXT,
        container_id TEXT,
        removal_date TEXT,
        reason TEXT,
        PRIMARY KEY (item_id, container_id)
    )
    """)
    
    # Ensure depleted_items table exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS depleted_items (
        item_id TEXT PRIMARY KEY,
        depletion_date TEXT
    )
    """)
    
    # Create setaside_useful table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS setaside_useful (
        item_id TEXT,
        original_container_id TEXT,
        temp_container_id TEXT,
        setaside_date TEXT,
        PRIMARY KEY (item_id)
    )
    """)
    
    conn.commit()
    conn.close()

def identify_waste_items() -> List[Dict[str, Any]]:
    """
    Identify items that are waste (expired, used up, or depleted)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get current date for expiry comparison
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Find expired items
        expired_query = """
        SELECT i.item_id, i.name, p.container_id, p.start_width, p.start_depth, p.start_height, 
               p.end_width, p.end_depth, p.end_height, 'Expired' as reason
        FROM items i
        JOIN placements p ON i.item_id = p.item_id
        WHERE i.expiry_date < ? AND i.expiry_date != ''
        """
        
        
        # Find items marked as depleted in the depleted_items table
        depleted_items_query = """
        SELECT i.item_id, i.name, p.container_id, p.start_width, p.start_depth, p.start_height, 
               p.end_width, p.end_depth, p.end_height, 'Depleted' as reason
        FROM items i
        JOIN placements p ON i.item_id = p.item_id
        JOIN depleted_items d ON i.item_id = d.item_id
        WHERE d.depletion_date != ''
        """
        
        # Execute queries
        cursor.execute(expired_query, (current_date,))
        expired_items = cursor.fetchall()
        
        cursor.execute(depleted_items_query)
        depleted_items = cursor.fetchall()
        
        # Combine results
        waste_items = []
        
        # Process all waste items
        for item in expired_items + depleted_items:
            waste_items.append({
                "itemId": item['item_id'],
                "name": item['name'],
                "reason": item['reason'],
                "containerId": item['container_id'],
                "position": {
                    "startCoordinates": {
                        "width": item['start_width'],
                        "depth": item['start_depth'],
                        "height": item['start_height']
                    },
                    "endCoordinates": {
                        "width": item['end_width'],
                        "depth": item['end_depth'],
                        "height": item['end_height']
                    }
                }
            })
        
        return waste_items
    
    except Exception as e:
        # Log the error
        print(f"Error in identify_waste_items: {str(e)}")
        raise
    
    finally:
        conn.close()

def get_item_details(item_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about an item"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
        SELECT i.*, p.container_id, p.start_width, p.start_depth, p.start_height, 
               p.end_width, p.end_depth, p.end_height
        FROM items i
        JOIN placements p ON i.item_id = p.item_id
        WHERE i.item_id = ?
        """, (item_id,))
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        item = dict(row)
        return {
            "itemId": item["item_id"],
            "name": item["name"],
            "width": item["width"],
            "depth": item["depth"],
            "height": item["height"],
            "mass": item["mass"],
            "containerId": item["container_id"],
            "position": {
                "startCoordinates": {
                    "width": item["start_width"],
                    "depth": item["start_depth"],
                    "height": item["start_height"]
                },
                "endCoordinates": {
                    "width": item["end_width"],
                    "depth": item["end_depth"],
                    "height": item["end_height"]
                }
            }
        }
    
    except Exception as e:
        # Log the error
        print(f"Error in get_item_details: {str(e)}")
        raise
    
    finally:
        conn.close()

def calculate_item_volume(item: Dict[str, Any]) -> float:
    """Calculate the volume of an item"""
    try:
        width = float(item["width"])
        depth = float(item["depth"])
        height = float(item["height"])
        return width * depth * height
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error calculating item volume: {str(e)}")
        return 0.0

def calculate_retrieval_steps(item_details: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Calculate retrieval steps for an item in a zero-gravity environment.
    Only considers items directly blocking access along the z-axis.
    Step numbering starts from 0.
    
    Args:
        item_details: Dictionary containing details of the item to retrieve
        
    Returns:
        List of retrieval steps with all necessary operations
    """
    try:
        if not item_details.get("containerId"):
            # For items without placement, add a simple retrieval step
            return [{
                "step": 0,
                "action": "retrieve",
                "itemId": item_details["itemId"],
                "itemName": item_details["name"]
            }]
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all items in the same container
        cursor.execute("""
            SELECT i.item_id, i.name, p.*
            FROM items i
            JOIN placements p ON i.item_id = p.item_id
            WHERE p.container_id = ? AND i.item_id != ?
        """, (item_details["containerId"], item_details["itemId"]))
        
        all_items = cursor.fetchall()
        conn.close()
        
        # Find items that block direct access along the z-axis
        blocking_items = []
        target_pos = item_details["position"]
        
        # Define the projection of the target item onto the front plane
        target_min_width = target_pos["startCoordinates"]["width"]
        target_max_width = target_pos["endCoordinates"]["width"]
        target_min_height = target_pos["startCoordinates"]["height"]
        target_max_height = target_pos["endCoordinates"]["height"]
        target_depth = target_pos["startCoordinates"]["depth"]  # Z-coordinate
        
        for item in all_items:
            item_dict = dict(item)
            
            # Check if this item overlaps with the target's projection on the front plane
            # and is positioned between the front and the target item
            
            # Width overlap check
            width_overlap = max(0, min(item_dict["end_width"], target_max_width) -
                              max(item_dict["start_width"], target_min_width)) > 0
            
            # Height overlap check
            height_overlap = max(0, min(item_dict["end_height"], target_max_height) -
                               max(item_dict["start_height"], target_min_height)) > 0
            
            # Check if the item is between the front plane and the target item
            is_blocking = (width_overlap and
                          height_overlap and
                          item_dict["end_depth"] <= target_depth and
                          item_dict["start_depth"] >= 0)
            
            if is_blocking:
                blocking_items.append(item_dict)
        
        # If no blocking items, the target is directly accessible
        if not blocking_items:
            return [{
                "step": 0,
                "action": "retrieve",
                "itemId": item_details["itemId"],
                "itemName": item_details["name"]
            }]
        
        # Sort blocking items by depth (from front to back)
        blocking_items.sort(key=lambda x: x["start_depth"])
        
        # Generate retrieval steps
        retrieval_steps = []
        
        # Add setAside steps
        for i, item in enumerate(blocking_items):
            retrieval_steps.append({
                "step": i,  # Step numbering starts from 0
                "action": "setAside",
                "itemId": item["item_id"],
                "itemName": item["name"]
            })
        
        # Add the retrieve step
        retrieval_steps.append({
            "step": len(blocking_items),  # This is after all setAside steps
            "action": "retrieve",
            "itemId": item_details["itemId"],
            "itemName": item_details["name"]
        })
        
        # Add placeBack steps (in reverse order)
        for i, item in enumerate(reversed(blocking_items)):
            retrieval_steps.append({
                "step": len(blocking_items) + 1 + i,
                "action": "placeBack",
                "itemId": item["item_id"],
                "itemName": item["name"]
            })
        
        return retrieval_steps
    
    except Exception as e:
        print(f"Error in calculate_retrieval_steps: {str(e)}")
        # Return a basic retrieval step as fallback
        return [{
            "step": 0,
            "action": "retrieve",
            "itemId": item_details["itemId"],
            "itemName": item_details["name"]
        }]

def create_return_plan(undocking_container_id: str, undocking_date: str, max_weight: float) -> Dict[str, Any]:
    """
    Create a plan for returning waste items to the undocking container
    """
    try:
        # Get all waste items
        waste_items = identify_waste_items()
        
        # Get detailed information about each waste item
        items_with_details = []
        total_weight = 0
        total_volume = 0
        
        for item in waste_items:
            details = get_item_details(item["itemId"])
            if details:
                # Add reason from waste identification
                details["reason"] = item["reason"]
                items_with_details.append(details)
                
                # Calculate weight and volume
                total_weight += float(details["mass"])
                total_volume += calculate_item_volume(details)
        
        # Sort items by priority and expiry date
        items_with_details.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        # Create return plan steps
        return_plan = []
        combined_retrieval_steps = []
        return_items = []
        step_counter = 1
        
        # Check if we exceed max weight
        if total_weight > max_weight:
            # Sort items by weight/volume efficiency
            items_with_details.sort(key=lambda x: float(x["mass"]) / max(0.001, calculate_item_volume(x)))
            
            # Reset counters
            total_weight = 0
            total_volume = 0
            
            # Add items until we reach max weight
            filtered_items = []
            for item in items_with_details:
                if total_weight + float(item["mass"]) <= max_weight:
                    filtered_items.append(item)
                    total_weight += float(item["mass"])
                    total_volume += calculate_item_volume(item)
                else:
                    # We've reached the weight limit
                    break
            
            items_with_details = filtered_items
        
        # Process items for the return plan
        for item in items_with_details:
            # Add to return plan steps
            return_plan.append({
                "step": step_counter,
                "itemId": item["itemId"],
                "itemName": item["name"],
                "fromContainer": item["containerId"],
                "toContainer": undocking_container_id
            })
            step_counter += 1
            
            # Calculate retrieval steps for this item
            item_retrieval_steps = calculate_retrieval_steps(item)
            
            # Adjust step numbers to ensure continuity across all items
            if combined_retrieval_steps:
                last_step = combined_retrieval_steps[-1]["step"]
                item_retrieval_steps = [
                    {**step, "step": step["step"] + last_step + 1}
                    for step in item_retrieval_steps
                ]
            
            combined_retrieval_steps.extend(item_retrieval_steps)
            
            # Add to return items list
            return_items.append({
                "itemId": item["itemId"],
                "name": item["name"],
                "reason": item["reason"]
            })
        
        # Create return manifest
        return_manifest = {
            "undockingContainerId": undocking_container_id,
            "undockingDate": undocking_date,
            "returnItems": return_items,
            "totalVolume": total_volume,
            "totalWeight": total_weight
        }
        
        return {
            "success": True,
            "returnPlan": return_plan,
            "retrievalSteps": combined_retrieval_steps,
            "returnManifest": return_manifest
        }
    
    except Exception as e:
        print(f"Error in create_return_plan: {str(e)}")
        raise

def complete_undocking(undocking_container_id: str, timestamp: str) -> Dict[str, Any]:
    """
    Mark items in the undocking container as removed from the system,
    record waste items in the database, and delete the container
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Create waste_log table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS waste_log (
            item_id TEXT,
            container_id TEXT,
            removal_date TEXT,
            reason TEXT,
            PRIMARY KEY (item_id, container_id)
        )
        """)
        
        # Get all items in the undocking container with their waste status
        cursor.execute("""
        SELECT p.item_id, i.name,
               CASE
                   WHEN i.expiry_date < datetime('now') AND i.expiry_date != '' THEN 'Expired'
                   WHEN d.item_id IS NOT NULL THEN 'Depleted'
                   ELSE NULL
               END as waste_reason
        FROM placements p
        JOIN items i ON p.item_id = i.item_id
        LEFT JOIN depleted_items d ON i.item_id = d.item_id
        WHERE p.container_id = ?
        """, (undocking_container_id,))
        
        items = cursor.fetchall()
        
        # Keep track of waste item IDs
        waste_item_ids = []
        
        # Record waste items in the waste_log
        for item in items:
            item_id = item['item_id']
            waste_reason = item['waste_reason']
            
            if waste_reason:  # This is a waste item
                cursor.execute("""
                INSERT OR REPLACE INTO waste_log (item_id, container_id, removal_date, reason)
                VALUES (?, ?, ?, ?)
                """, (item_id, undocking_container_id, timestamp, waste_reason))
                waste_item_ids.append(item_id)
        
        # Remove items from placements
        cursor.execute("""
        DELETE FROM placements
        WHERE container_id = ?
        """, (undocking_container_id,))
        
        # Delete waste items from items table
        if waste_item_ids:
            # Convert list to string for SQL IN clause
            item_ids_str = ','.join(f"'{item_id}'" for item_id in waste_item_ids)
            cursor.execute(f"""
            DELETE FROM items
            WHERE item_id IN ({item_ids_str})
            """)
        
        # Delete the container from containers table
        cursor.execute("""
        DELETE FROM containers
        WHERE container_id = ?
        """, (undocking_container_id,))
        
        conn.commit()
        
        return {
            "success": True,
            "itemsRemoved": len(items),
            "wasteItemsDeleted": len(waste_item_ids),
            "containerDeleted": undocking_container_id
        }
    
    except Exception as e:
        conn.rollback()
        print(f"Error in complete_undocking: {str(e)}")
        raise
    
    finally:
        conn.close()

@router.get("/identify", response_model=WasteIdentifyResponse)
async def identify_waste():
    """
    Identify waste items (expired or used up) in the system
    """
    try:
        # Ensure database setup
        setup_database()
        
        waste_items = identify_waste_items()
        return {"success": True, "wasteItems": waste_items}
    except Exception as e:
        print(f"API Error in identify_waste: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/return-plan", response_model=ReturnPlanResponse)
async def get_return_plan(request: ReturnPlanRequest):
    """
    Create a plan for returning waste items to the undocking container
    """
    try:
        # Ensure database setup
        setup_database()
        
        plan = create_return_plan(
            request.undockingContainerId,
            request.undockingDate,
            request.maxWeight
        )
        return plan
    except Exception as e:
        print(f"API Error in get_return_plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/complete-undocking", response_model=UndockingResponse)
async def complete_undocking_process(request: UndockingRequest):
    """
    Mark items in the undocking container as removed from the system
    """
    try:
        # Ensure database setup
        setup_database()
        
        result = complete_undocking(
            request.undockingContainerId,
            request.timestamp
        )
        return result
    except Exception as e:
        print(f"API Error in complete_undocking_process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize database tables when module loads
setup_database()