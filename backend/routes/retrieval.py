import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import time

# Import your database module
from database import get_all_items, get_all_containers, get_all_placements

router = APIRouter()

# Define Pydantic models for request/response validation
class Coordinates(BaseModel):
    width: float
    depth: float 
    height: float

class Position(BaseModel):
    startCoordinates: Coordinates
    endCoordinates: Coordinates

class Item(BaseModel):
    itemId: str
    name: str
    containerId: Optional[str] = None
    zone: Optional[str] = None
    position: Optional[Position] = None
    usage_limit: Optional[int] = None
    current_usage: Optional[int] = None

class RetrievalStep(BaseModel):
    step: int
    action: str  # "remove", "setAside", "retrieve", "placeBack"
    itemId: str
    itemName: str

class SearchResponse(BaseModel):
    success: bool
    found: bool
    item: Optional[Item] = None
    retrievalSteps: List[RetrievalStep] = []

class RetrieveRequest(BaseModel):
    itemId: str
    userId: str
    timestamp: str  # ISO format

class PlaceBackRequest(BaseModel):
    itemId: str
    userId: str
    timestamp: str  # ISO format
    containerId: str
    position: Position
    originalPosition: Optional[Position] = None  # To track if coordinates changed

class BasicResponse(BaseModel):
    success: bool

# Ensure database setup on module import
def ensure_database_setup():
    """
    Ensure all necessary database tables are created
    """
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    # Create items table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS items (
        item_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        category TEXT,
        tags TEXT,
        expiry_date TEXT,
        priority INTEGER DEFAULT 0,
        usage_limit INTEGER,
        current_usage INTEGER DEFAULT 0,
        preferred_zone TEXT
    )
    """)
    
    # Create containers table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS containers (
        container_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        zone TEXT,
        max_width REAL,
        max_depth REAL,
        max_height REAL
    )
    """)
    
    # Create placements table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS placements (
        placement_id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_id TEXT,
        container_id TEXT,
        start_width REAL,
        start_depth REAL,
        start_height REAL,
        end_width REAL,
        end_depth REAL,
        end_height REAL,
        rotated INTEGER DEFAULT 0,
        timestamp TEXT,
        FOREIGN KEY(item_id) REFERENCES items(item_id),
        FOREIGN KEY(container_id) REFERENCES containers(container_id)
    )
    """)
    
    # Create usage_log table with new fields if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usage_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_id TEXT,
        user_id TEXT,
        timestamp TEXT,
        container_id TEXT,
        from_container TEXT,
        to_container TEXT,
        reason TEXT,
        FOREIGN KEY(item_id) REFERENCES items(item_id),
        FOREIGN KEY(container_id) REFERENCES containers(container_id),
        FOREIGN KEY(from_container) REFERENCES containers(container_id),
        FOREIGN KEY(to_container) REFERENCES containers(container_id)
    )
    """)
    
    # Create depleted_items table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS depleted_items (
        item_id TEXT PRIMARY KEY,
        depletion_date TEXT,
        FOREIGN KEY(item_id) REFERENCES items(item_id)
    )
    """)
    
    conn.commit()
    conn.close()

# Call the function to ensure tables exist
ensure_database_setup()

def get_item_details(item_id: str = None, item_name: str = None) -> Dict[str, Any]:
    """
    Retrieve item details from the database by ID or name
    
    Args:
        item_id: The unique identifier of the item to retrieve
        item_name: The name of the item to retrieve
        
    Returns:
        Dictionary containing item details or None if not found
    """
    conn = sqlite3.connect('inventory.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # First check if the item exists at all
    if item_id:
        cursor.execute("SELECT * FROM items WHERE item_id = ?", (item_id,))
    elif item_name:
        cursor.execute("SELECT * FROM items WHERE name = ? COLLATE NOCASE", (item_name,))
    else:
        conn.close()
        return None
    
    item_row = cursor.fetchone()
    
    if not item_row:
        conn.close()
        return None
    
    item_dict = dict(item_row)
    item_id = item_dict["item_id"]
    
    # Now get placement information if it exists
    cursor.execute("""
        SELECT p.*, c.zone
        FROM placements p
        JOIN containers c ON p.container_id = c.container_id
        WHERE p.item_id = ?
    """, (item_id,))
    
    placement_row = cursor.fetchone()
    conn.close()
    
    # If no placement found, return basic item info with empty position
    if not placement_row:
        return {
            "itemId": item_dict["item_id"],
            "name": item_dict["name"],
            "containerId": None,
            "zone": item_dict.get("preferred_zone", None),
            "position": {
                "startCoordinates": {
                    "width": 0,
                    "depth": 0,
                    "height": 0
                },
                "endCoordinates": {
                    "width": 0,
                    "depth": 0,
                    "height": 0
                }
            },
            "usage_limit": item_dict.get("usage_limit"),
            "current_usage": item_dict.get("current_usage", 0)
        }
    
    # We have both item and placement info
    placement_dict = dict(placement_row)
    
    # Format the response according to the required structure
    item_details = {
        "itemId": item_dict["item_id"],
        "name": item_dict["name"],
        "containerId": placement_dict["container_id"],
        "zone": placement_dict["zone"],
        "position": {
            "startCoordinates": {
                "width": placement_dict["start_width"],
                "depth": placement_dict["start_depth"],
                "height": placement_dict["start_height"]
            },
            "endCoordinates": {
                "width": placement_dict["end_width"],
                "depth": placement_dict["end_depth"],
                "height": placement_dict["end_height"]
            }
        },
        "usage_limit": item_dict.get("usage_limit"),
        "current_usage": item_dict.get("current_usage", 0)
    }
    
    return item_details

def calculate_retrieval_steps(item_details: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Calculate retrieval steps for an item in a zero-gravity environment.
    Only considers items directly blocking access along the z-axis.
    Only setAside operations are included in the steps.
    Step numbering starts from 0.
    
    Args:
        item_details: Dictionary containing details of the item to retrieve
        
    Returns:
        List of retrieval steps with only necessary operations
    """
    if not item_details["containerId"]:
        # For items without placement, add a simple retrieval step
        return [{
            "step": 0,
            "action": "retrieve",
            "itemId": item_details["itemId"],
            "itemName": item_details["name"]
        }]
    
    conn = sqlite3.connect('inventory.db')
    conn.row_factory = sqlite3.Row
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
    
    # Generate retrieval steps - only include setAside and retrieve actions
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
    
    return retrieval_steps

def update_item_usage(item_id: str, user_id: str, timestamp: str) -> bool:
    """
    Update the item usage count in the database with retry logic
    
    Args:
        item_id: ID of the item
        user_id: ID of the user
        timestamp: ISO formatted timestamp
    
    Returns:
        True if successful, False otherwise
    """
    max_retries = 5
    retry_delay = 0.5  # seconds
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect('inventory.db', timeout=20.0)
            cursor = conn.cursor()
            
            # Check if the item exists and has a usage limit
            cursor.execute("""
                SELECT usage_limit, current_usage 
                FROM items 
                WHERE item_id = ?
            """, (item_id,))
            
            result = cursor.fetchone()
            if not result:
                print(f"Item {item_id} not found in database")
                conn.close()
                return False
            
            usage_limit, current_usage = result
            
            # Initialize current_usage if None
            if current_usage is None:
                current_usage = 0
            
            # If there's a usage limit, check if already depleted
            if usage_limit is not None and current_usage >= usage_limit:
                print(f"Item {item_id} is already depleted")
                conn.close()
                return False
            
            # Increment usage count in items table
            new_usage = current_usage + 1
            try:
                cursor.execute("""
                UPDATE items 
                SET current_usage = ? 
                WHERE item_id = ?
                """, (new_usage, item_id))
            except Exception as e:
                print(f"Error updating current_usage: {str(e)}")
                conn.rollback()
                conn.close()
                return False
            
            # If usage limit is specified and this usage reached the limit, mark as depleted
            if usage_limit is not None and new_usage >= usage_limit:
                # Create depleted_items table if it doesn't exist
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS depleted_items (
                    item_id TEXT PRIMARY KEY,
                    depletion_date TEXT,
                    FOREIGN KEY(item_id) REFERENCES items(item_id)
                )
                """)
                
                # Mark as depleted
                current_time = datetime.now().isoformat()
                try:
                    cursor.execute("""
                    INSERT OR REPLACE INTO depleted_items (item_id, depletion_date)
                    VALUES (?, ?)
                    """, (item_id, current_time))
                except Exception as e:
                    print(f"Error inserting into depleted_items: {str(e)}")
                    conn.rollback()
                    conn.close()
                    return False
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                print(f"Database locked, retrying in {retry_delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                print(f"Database error after {attempt+1} attempts: {str(e)}")
                return False
        except Exception as e:
            print(f"Critical error in update_item_usage: {str(e)}")
            return False
        
def log_item_usage(item_id: str, user_id: str, timestamp: str, container_id: str = None, 
                   from_container: str = None, to_container: str = None, reason: str = None) -> bool:
    """
    Log the item usage in the database
    
    Args:
        item_id: ID of the item
        user_id: ID of the user
        timestamp: ISO formatted timestamp
        container_id: ID of the current container
        from_container: ID of the source container
        to_container: ID of the destination container
        reason: Reason for movement (retrieval, rearrangement)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect('inventory.db')
        cursor = conn.cursor()
        
        # Create usage_log table if it doesn't exist with new fields
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id TEXT,
            user_id TEXT,
            timestamp TEXT,
            container_id TEXT,
            from_container TEXT,
            to_container TEXT,
            reason TEXT,
            FOREIGN KEY(item_id) REFERENCES items(item_id),
            FOREIGN KEY(container_id) REFERENCES containers(container_id),
            FOREIGN KEY(from_container) REFERENCES containers(container_id),
            FOREIGN KEY(to_container) REFERENCES containers(container_id)
        )
        """)
        
        # Insert usage log with container_id and new fields
        cursor.execute("""
        INSERT INTO usage_log (item_id, user_id, timestamp, container_id, from_container, to_container, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (item_id, user_id, timestamp, container_id, from_container, to_container, reason))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error logging item usage: {str(e)}")
        return False

def update_item_placement(place_request: PlaceBackRequest) -> bool:
    """
    Update the placement of an item in the database
    
    Args:
        place_request: The placement request details
        
    Returns:
        True if successful, False otherwise
    """
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    # Check if the item exists
    cursor.execute("SELECT 1 FROM items WHERE item_id = ?", (place_request.itemId,))
    if not cursor.fetchone():
        conn.close()
        return False
    
    # Check if the container exists
    cursor.execute("SELECT 1 FROM containers WHERE container_id = ?", (place_request.containerId,))
    if not cursor.fetchone():
        conn.close()
        return False
    
    # Update the placement
    cursor.execute("""
    INSERT OR REPLACE INTO placements 
    (item_id, container_id, start_width, start_depth, start_height, end_width, end_depth, end_height, rotated, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        place_request.itemId,
        place_request.containerId,
        place_request.position.startCoordinates.width,
        place_request.position.startCoordinates.depth,
        place_request.position.startCoordinates.height,
        place_request.position.endCoordinates.width,
        place_request.position.endCoordinates.depth,
        place_request.position.endCoordinates.height,
        0,  # Not rotating by default
        place_request.timestamp
    ))
    
    conn.commit()
    conn.close()
    return True

def ensure_items_table():
    """
    Ensure items table exists with required columns
    """
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    # Check if items table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='items'")
    if not cursor.fetchone():
        # Create items table
        cursor.execute("""
        CREATE TABLE items (
            item_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            category TEXT,
            tags TEXT,
            expiry_date TEXT,
            priority INTEGER DEFAULT 0,
            usage_limit INTEGER,
            current_usage INTEGER DEFAULT 0,
            preferred_zone TEXT
        )
        """)
        conn.commit()
    else:
        # Check if required columns exist
        cursor.execute("PRAGMA table_info(items)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # Add missing columns if needed
        if "usage_limit" not in column_names:
            cursor.execute("ALTER TABLE items ADD COLUMN usage_limit INTEGER")
        
        if "current_usage" not in column_names:
            cursor.execute("ALTER TABLE items ADD COLUMN current_usage INTEGER DEFAULT 0")
        
        conn.commit()
    
    conn.close()

# Call this function before any operations
ensure_items_table()

@router.get("/search", response_model=SearchResponse)
async def search_item(
    itemId: str = Query(None, description="Item ID to search for"),
    itemName: str = Query(None, description="Item name to search for"),
    userId: str = Query(None, description="User ID for tracking purposes")
):
    """
    Search for an item by ID or name and get retrieval instructions
    
    Args:
        itemId: The ID of the item to search for
        itemName: The name of the item to search for
        userId: The ID of the user making the request
        
    Returns:
        SearchResponse: Item details and retrieval steps
    """
    if not itemId and not itemName:
        raise HTTPException(status_code=400, detail="Either itemId or itemName must be provided")
    
    try:
        # If both are provided, prioritize itemId
        if itemId:
            item_details = get_item_details(item_id=itemId)
        else:
            item_details = get_item_details(item_name=itemName)
        
        # If not found by exact match, try searching for items with similar names
        if not item_details and itemName and not itemId:
            conn = sqlite3.connect('inventory.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Search for partial name matches
            cursor.execute("""
                SELECT item_id 
                FROM items 
                WHERE name LIKE ? COLLATE NOCASE
                LIMIT 1
            """, (f"%{itemName}%",))
            
            similar_item = cursor.fetchone()
            conn.close()
            
            if similar_item:
                # Try again with the similar item ID
                item_details = get_item_details(item_id=similar_item["item_id"])
        
        if not item_details:
            # No items found
            return SearchResponse(
                success=True,
                found=False
            )
        
        # Calculate retrieval steps
        retrieval_steps = calculate_retrieval_steps(item_details)
        
        return SearchResponse(
            success=True,
            found=True,
            item=item_details,
            retrievalSteps=retrieval_steps
        )
    
    except Exception as e:
        # Log the error
        print(f"Error in search_item: {str(e)}")
        # Return a proper error response
        return SearchResponse(
            success=False,
            found=False,
            retrievalSteps=[]
        )

@router.post("/retrieve", response_model=BasicResponse)
async def retrieve_item(request: RetrieveRequest):
    """
    Record item retrieval and update usage count
    """
    try:
        # First check if item exists in the database directly
        conn = sqlite3.connect('inventory.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if item exists
        cursor.execute("SELECT item_id, usage_limit, current_usage FROM items WHERE item_id = ?", (request.itemId,))
        item_row = cursor.fetchone()
        
        if not item_row:
            conn.close()
            raise HTTPException(status_code=404, detail="Item not found")
        
        item_dict = dict(item_row)
        
        # Check usage limits
        usage_limit = item_dict.get("usage_limit")
        current_usage = item_dict.get("current_usage", 0)
        
        if current_usage is None:
            current_usage = 0

        if usage_limit is not None and current_usage >= usage_limit:
            raise HTTPException(status_code=400, detail="Item is depleted and cannot be used")
        
        # Update item usage count in the database - but don't log the activity yet
        success = update_item_usage(request.itemId, request.userId, request.timestamp)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update item usage count")
            
        return BasicResponse(success=True)
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the error
        error_msg = f"Error in retrieve_item: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/place", response_model=BasicResponse)
async def place_back_item(request: PlaceBackRequest):
    """
    Record item placement back into storage and log the activity
    
    Args:
        request: The place-back request containing item and position details
        
    Returns:
        BasicResponse: Success status
    """
    try:
        # First, get the current container and position of the item (if any)
        conn = sqlite3.connect('inventory.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get current placement details
        cursor.execute("""
            SELECT container_id, start_width, start_depth, start_height, end_width, end_depth, end_height
            FROM placements 
            WHERE item_id = ?
        """, (request.itemId,))
        
        placement_row = cursor.fetchone()
        
        # Get the origin container
        from_container = None
        original_position = None
        if placement_row:
            from_container = placement_row["container_id"]
            original_position = {
                "startCoordinates": {
                    "width": placement_row["start_width"],
                    "depth": placement_row["start_depth"],
                    "height": placement_row["start_height"]
                },
                "endCoordinates": {
                    "width": placement_row["end_width"],
                    "depth": placement_row["end_depth"],
                    "height": placement_row["end_height"]
                }
            }
        
        conn.close()
        
        # Determine if this is a retrieval or rearrangement
        reason = "retrieval"  # Default reason
        
        # If containers are different, it's a rearrangement
        if from_container != request.containerId:
            reason = "rearrangement"
        # If containers are the same but position changed, it's also a rearrangement
        elif original_position:
            # Check if the position changed
            new_pos = request.position
            orig_pos = Position(**original_position)
            
            # Compare positions
            if (new_pos.startCoordinates.width != orig_pos.startCoordinates.width or
                new_pos.startCoordinates.depth != orig_pos.startCoordinates.depth or
                new_pos.startCoordinates.height != orig_pos.startCoordinates.height or
                new_pos.endCoordinates.width != orig_pos.endCoordinates.width or
                new_pos.endCoordinates.depth != orig_pos.endCoordinates.depth or
                new_pos.endCoordinates.height != orig_pos.endCoordinates.height):
                reason = "rearrangement"
        
        # Update item placement in the database
        success = update_item_placement(request)
        
        if not success:
            raise HTTPException(
                status_code=404, 
                detail="Failed to place item back. Item or container not found."
            )
        
        # Log the complete activity with appropriate reason
        log_success = log_item_usage(
            request.itemId, 
            request.userId, 
            request.timestamp, 
            request.containerId,
            from_container=from_container,
            to_container=request.containerId,
            reason=reason
        )
        
        if not log_success:
            # This is worth noting but not critical for the operation
            print(f"Warning: Failed to log usage for item {request.itemId}")
        
        return BasicResponse(success=True)
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the error
        error_msg = f"Error in place_back_item: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/debug/item/{item_id}")
async def debug_item(item_id: str):
    """
    Debug endpoint to check the raw database entries for an item
    """
    conn = sqlite3.connect('inventory.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check items table
    cursor.execute("SELECT * FROM items WHERE item_id = ?", (item_id,))
    item_row = cursor.fetchone()
    
    # Check placements table
    cursor.execute("SELECT * FROM placements WHERE item_id = ?", (item_id,))
    placement_row = cursor.fetchone()
    
    # Get container info if placement exists
    container_row = None
    if placement_row:
        cursor.execute("SELECT * FROM containers WHERE container_id = ?", (placement_row["container_id"],))
        container_row = cursor.fetchone()
    
    conn.close()
    
    return {
        "item_exists": item_row is not None,
        "item_data": dict(item_row) if item_row else None,
        "placement_exists": placement_row is not None,
        "placement_data": dict(placement_row) if placement_row else None,
        "container_exists": container_row is not None,
        "container_data": dict(container_row) if container_row else None
    }

@router.get("/debug/database-structure")
async def debug_database_structure():
    """
    Debug endpoint to check the database structure
    """
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    result = {}
    
    # For each table, get its structure
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Format column information
        column_info = []
        for col in columns:
            column_info.append({
                "id": col[0],
                "name": col[1],
                "type": col[2],
                "notnull": col[3],
                "default": col[4],
                "pk": col[5]
            })
        
        result[table_name] = column_info
    
    conn.close()
    return result

@router.post("/debug/create-test-item")
async def create_test_item():
    """
    Create a test item with usage limits for testing
    """
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    # Create a test item
    test_item_id = f"test-item-{int(time.time())}"
    
    try:
        cursor.execute("""
        INSERT INTO items (item_id, name, usage_limit, current_usage)
        VALUES (?, ?, ?, ?)
        """, (test_item_id, f"Test Item {test_item_id}", 5, 0))
        
        # Create a test container if needed
        cursor.execute("SELECT 1 FROM containers WHERE container_id = 'test-container' LIMIT 1")
        if not cursor.fetchone():
            cursor.execute("""
            INSERT INTO containers (container_id, name, zone, max_width, max_depth, max_height)
            VALUES (?, ?, ?, ?, ?, ?)
            """, ('test-container', 'Test Container', 'A', 100.0, 100.0, 100.0))
        
        # Create a placement for the test item
        cursor.execute("""
        INSERT INTO placements (item_id, container_id, start_width, start_depth, start_height, 
                                end_width, end_depth, end_height, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (test_item_id, 'test-container', 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, datetime.now().isoformat()))
        
        conn.commit()
        
        return {
            "success": True,
            "itemId": test_item_id,
            "message": "Test item created successfully"
        }
    except Exception as e:
        conn.rollback()
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        conn.close()