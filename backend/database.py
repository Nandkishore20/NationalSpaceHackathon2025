import sqlite3
import csv
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Create a router for database-related routes
router = APIRouter()

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('/app/backend/inventory.db')
    cursor = conn.cursor()
    
    # Create Items table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS items (
        item_id TEXT PRIMARY KEY,
        name TEXT,
        width REAL,
        depth REAL,
        height REAL,
        mass REAL,
        priority INTEGER,
        expiry_date TEXT,
        usage_limit INTEGER,
        preferred_zone TEXT
    )
    ''')
    
    # Create Containers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS containers (
        zone TEXT,
        container_id TEXT PRIMARY KEY,
        width REAL,
        depth REAL,
        height REAL
    )
    ''')
    
    # Create Placements table to store occupied spaces
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS placements (
        item_id TEXT PRIMARY KEY,
        container_id TEXT,
        start_width REAL,
        start_depth REAL,
        start_height REAL,
        end_width REAL,
        end_depth REAL,
        end_height REAL,
        rotated INTEGER,
        timestamp TEXT,
        FOREIGN KEY(item_id) REFERENCES items(item_id),
        FOREIGN KEY(container_id) REFERENCES containers(container_id)
    )
    ''')
    
    conn.commit()
    conn.close()

def insert_or_update_items(items_data: List[Dict[str, Any]]):
    """Insert or update items in the database"""
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    for item in items_data:
        cursor.execute('''
        INSERT OR REPLACE INTO items 
        (item_id, name, width, depth, height, mass, priority, expiry_date, usage_limit, preferred_zone)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item['item_id'], 
            item['name'], 
            float(item['width']), 
            float(item['depth']), 
            float(item['height']), 
            float(item['mass']),
            int(item['priority']), 
            item['expiry_date'], 
            int(item['usage_limit']) if item['usage_limit'] else None, 
            item['preferred_zone']
        ))
    
    conn.commit()
    conn.close()

def insert_or_update_containers(containers_data: List[Dict[str, Any]]):
    """Insert or update containers in the database"""
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    for container in containers_data:
        cursor.execute('''
        INSERT OR REPLACE INTO containers 
        (zone, container_id, width, depth, height)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            container['zone'],
            container['container_id'], 
            float(container['width']), 
            float(container['depth']), 
            float(container['height'])
        ))
    
    conn.commit()
    conn.close()

def insert_placement(placement: Dict[str, Any]):
    """
    Insert a placement record into the database.
    Expected keys in placement dict:
    itemId, containerId, position (with startCoordinates and endCoordinates), rotated.
    """
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    pos = placement["position"]
    start = pos["startCoordinates"]
    end = pos["endCoordinates"]
    rotated = 1 if placement.get("rotated", False) else 0
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
    INSERT OR REPLACE INTO placements 
    (item_id, container_id, start_width, start_depth, start_height, end_width, end_depth, end_height, rotated, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        placement["itemId"],
        placement["containerId"],
        start["width"],
        start["depth"],
        start["height"],
        end["width"],
        end["depth"],
        end["height"],
        rotated,
        timestamp
    ))
    
    conn.commit()
    conn.close()

def get_all_placements() -> List[Dict[str, Any]]:
    """Retrieve all stored placements (occupied spaces) from the database."""
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM placements')
    rows = cursor.fetchall()
    conn.close()
    
    placements = []
    for row in rows:
        (item_id, container_id, start_width, start_depth, start_height, 
         end_width, end_depth, end_height, rotated, timestamp) = row
        placements.append({
            "itemId": item_id,
            "containerId": container_id,
            "position": {
                "startCoordinates": {
                    "width": start_width,
                    "depth": start_depth,
                    "height": start_height
                },
                "endCoordinates": {
                    "width": end_width,
                    "depth": end_depth,
                    "height": end_height
                }
            },
            "rotated": bool(rotated),
            "timestamp": timestamp
        })
    return placements

@router.post("/init-db")
def initialize_database():
    """Explicitly initialize database tables"""
    try:
        init_database()  # Call your existing initialization function
        return {"message": "Database tables created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Modify get_all_items() to match Pydantic field names
@router.get("/get/items")
def get_all_items():
    """Retrieve all items with field name mapping"""
    conn = sqlite3.connect('inventory.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM items")
    items = []
    for row in cursor.fetchall():
        item = dict(row)
        # Map database fields to Pydantic model names
        items.append({
            "itemId": item["item_id"],
            "name": item["name"],
            "width": item["width"],
            "depth": item["depth"],
            "height": item["height"],
            "mass": item["mass"],
            "priority": item["priority"],
            "expiryDate": item["expiry_date"],
            "usageLimit": item["usage_limit"],
            "preferredZone": item["preferred_zone"]
        })
    conn.close()
    return items

@router.get("/get/containers")
def get_all_containers():
    """Retrieve all containers with field name mapping"""
    conn = sqlite3.connect('inventory.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM containers")
    containers = []
    for row in cursor.fetchall():
        container = dict(row)
        containers.append({
            "containerId": container["container_id"],  # Map to Pydantic field
            "zone": container["zone"],
            "width": container["width"],
            "depth": container["depth"],
            "height": container["height"]
        })
    conn.close()
    return containers


@router.post("/import/items")
async def upload_items(items_file: UploadFile = File(...)):
    """Upload and process items CSV file"""
    try:
        # Read CSV file
        contents = await items_file.read()
        csv_contents = contents.decode('utf-8').splitlines()
        csv_reader = csv.DictReader(csv_contents)
        
        # Validate required columns
        required_columns = [
            'item_id', 'name', 'width', 'depth', 'height', 
            'mass', 'priority', 'expiry_date', 'usage_limit', 'preferred_zone'
        ]
        
        if not all(col in csv_reader.fieldnames for col in required_columns):
            raise HTTPException(status_code=400, detail="Missing required columns in items CSV")
        
        # Convert CSV data to list of dictionaries
        items_data = list(csv_reader)
        
        # Insert or update items
        insert_or_update_items(items_data)
        
        return JSONResponse(content={"message": f"Successfully uploaded {len(items_data)} items"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/import/containers")
async def import_containers(containers_file: UploadFile = File(...)):
    """Upload and process containers CSV file"""
    try:
        # Read CSV file
        contents = await containers_file.read()
        csv_contents = contents.decode('utf-8').splitlines()
        csv_reader = csv.DictReader(csv_contents)
        
        # Validate required columns
        required_columns = ['zone', 'container_id', 'width', 'depth', 'height']
        
        if not all(col in csv_reader.fieldnames for col in required_columns):
            raise HTTPException(status_code=400, detail="Missing required columns in containers CSV")
        
        # Convert CSV data to list of dictionaries
        containers_data = list(csv_reader)
        
        # Insert or update containers
        insert_or_update_containers(containers_data)
        
        return JSONResponse(content={"message": f"Successfully uploaded {len(containers_data)} containers"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Initialize the database when the module is imported
init_database()
