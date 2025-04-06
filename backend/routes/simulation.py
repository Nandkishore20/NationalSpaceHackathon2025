from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import sqlite3
from fastapi.responses import JSONResponse

router = APIRouter()

# Pydantic models
class ItemToUse(BaseModel):
    itemId: str
    name: Optional[str] = None

class SimulateDayRequest(BaseModel):
    numOfDays: Optional[int] = None
    toTimestamp: Optional[str] = None
    itemsToBeUsedPerDay: List[ItemToUse]
    
    def validate_time_params(self):
        # At least one time parameter must be provided
        if self.numOfDays is None and self.toTimestamp is None:
            raise ValueError("Either numOfDays or toTimestamp must be provided")
        return True

class ItemUsedChange(BaseModel):
    itemId: str
    name: str
    remainingUses: int

class ItemChange(BaseModel):
    itemId: str
    name: str

class Changes(BaseModel):
    itemsUsed: List[ItemUsedChange] = []
    itemsExpired: List[ItemChange] = []
    itemsDepletedToday: List[ItemChange] = []

class SimulateDayResponse(BaseModel):
    success: bool
    newDate: str
    changes: Changes

# Database functions
def get_items_from_db():
    conn = sqlite3.connect('inventory.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # First get basic item information
    cursor.execute("""
    SELECT
        item_id as itemId,
        name,
        expiry_date as expiryDate,
        usage_limit as usageLimit,
        CASE
            WHEN item_id IN (SELECT item_id FROM placements) THEN 1
            ELSE 0
        END as isPlaced
    FROM items
    """)
    
    items = [dict(row) for row in cursor.fetchall()]
    
    # For each item, get the current usage count from usage_log
    # The usage_log table has been updated with additional fields
    for item in items:
        cursor.execute("""
        SELECT COUNT(*) as usage_count
        FROM usage_log
        WHERE item_id = ?
        """, (item['itemId'],))
        
        usage_result = cursor.fetchone()
        used_count = usage_result['usage_count'] if usage_result else 0
        
        # Calculate remaining uses based on usage limit and actual usage
        if item['usageLimit'] is not None:
            item['remainingUses'] = max(0, item['usageLimit'] - used_count)
        else:
            item['remainingUses'] = 0
            
        item['isActive'] = item['remainingUses'] > 0 if item['usageLimit'] is not None else True
    
    conn.close()
    return items

# API endpoint
@router.post("/simulate/day", response_model=SimulateDayResponse)
async def simulate_day(request: SimulateDayRequest = Body(...)):
    try:
        # Validate request
        try:
            request.validate_time_params()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Date handling
        current_date = datetime.now()
        
        # Calculate target date - prioritize numOfDays if present
        if request.numOfDays is not None:
            if request.numOfDays <= 0:
                raise HTTPException(status_code=400, detail="numOfDays must be positive")
            target_date = current_date + timedelta(days=request.numOfDays)
            days_to_simulate = request.numOfDays
        else:  # Use toTimestamp
            try:
                if 'Z' in request.toTimestamp:
                    target_date = datetime.fromisoformat(request.toTimestamp.replace('Z', '+00:00'))
                else:
                    target_date = datetime.fromisoformat(request.toTimestamp)
                
                if target_date <= current_date:
                    raise HTTPException(status_code=400, detail="toTimestamp must be in the future")
                
                days_to_simulate = (target_date - current_date).days
                if days_to_simulate <= 0:
                    days_to_simulate = 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid timestamp format")

        # Get items with current usage counts and prepare data
        db_items = get_items_from_db()
        items_lookup = {item['itemId']: item for item in db_items}
        
        # Validate items exist
        for item in request.itemsToBeUsedPerDay:
            if item.itemId not in items_lookup:
                raise HTTPException(status_code=404, detail=f"Item {item.itemId} not found")
            
            # If the user provided a name, use it instead of the DB name
            if item.name:
                # Store the provided name in the lookup for this simulation
                items_lookup[item.itemId]['display_name'] = item.name
            else:
                # Use the DB name if not provided
                items_lookup[item.itemId]['display_name'] = items_lookup[item.itemId]['name']
        
        # Keep track of items with zero remaining uses at the start
        initially_depleted_items = []
        for item in request.itemsToBeUsedPerDay:
            item_id = item.itemId
            if item_id in items_lookup and items_lookup[item_id]['usageLimit'] is not None and items_lookup[item_id]['remainingUses'] == 0:
                initially_depleted_items.append(item_id)
        
        # Track changes by day
        daily_changes = []
        # Initialize with empty changes for each day
        for _ in range(days_to_simulate):
            daily_changes.append({
                "itemsUsed": [],
                "itemsExpired": [],
                "itemsDepletedToday": []
            })

        current_sim_date = current_date
        for day in range(days_to_simulate):
            current_sim_date += timedelta(days=1)
            
            # For each day, process all items
            for item_to_use in request.itemsToBeUsedPerDay:
                item_id = item_to_use.itemId
                item = items_lookup[item_id]
                
                if not item['isActive']:
                    continue
                
                # Check expiration
                if item['expiryDate']:
                    expiry_date = datetime.fromisoformat(item['expiryDate']).date()
                    if current_sim_date.date() > expiry_date:
                        if not any(i['itemId'] == item_id for i in daily_changes[day]['itemsExpired']):
                            daily_changes[day]['itemsExpired'].append({
                                "itemId": item_id,
                                "name": item['display_name']
                            })
                        item['isActive'] = False
                        continue
                
                # Process usage - decrease usage by 1 each day the item is used
                if item['usageLimit'] is not None:
                    if item['remainingUses'] > 0:
                        # Check if this use will deplete the item
                        if item['remainingUses'] == 1:
                            if not any(i['itemId'] == item_id for i in daily_changes[day]['itemsDepletedToday']):
                                daily_changes[day]['itemsDepletedToday'].append({
                                    "itemId": item_id,
                                    "name": item['display_name']
                                })
                            item['isActive'] = False
                        
                        # Decrement the remaining uses
                        item['remainingUses'] -= 1
                    
                # Update or add to the items_used list for this day
                existing = next((i for i in daily_changes[day]['itemsUsed'] if i['itemId'] == item_id), None)
                if existing:
                    existing['remainingUses'] = item['remainingUses']
                else:
                    daily_changes[day]['itemsUsed'].append({
                        "itemId": item_id,
                        "name": item['display_name'],
                        "remainingUses": item['remainingUses']
                    })

        # Consolidate changes for the final response
        items_used = []
        items_expired = []
        items_depleted_today = []
        
        # Add all requested items to the response with their final remainingUses
        for item in request.itemsToBeUsedPerDay:
            item_id = item.itemId
            db_item = items_lookup[item_id]
            
            items_used.append(ItemUsedChange(
                itemId=item_id,
                name=db_item['display_name'],
                remainingUses=db_item['remainingUses']
            ))
        
        # Add items that started with zero remaining uses to itemsDepletedToday
        for item_id in initially_depleted_items:
            if not any(i.itemId == item_id for i in items_depleted_today):
                items_depleted_today.append(ItemChange(
                    itemId=item_id,
                    name=items_lookup[item_id]['display_name']
                ))
        
        # Now collect all expired and depleted items across all days
        for day_changes in daily_changes:
            for expired_item in day_changes['itemsExpired']:
                if not any(i.itemId == expired_item['itemId'] for i in items_expired):
                    items_expired.append(ItemChange(
                        itemId=expired_item['itemId'],
                        name=expired_item['name']
                    ))
            
            for depleted_item in day_changes['itemsDepletedToday']:
                if not any(i.itemId == depleted_item['itemId'] for i in items_depleted_today):
                    items_depleted_today.append(ItemChange(
                        itemId=depleted_item['itemId'],
                        name=depleted_item['name']
                    ))

        return SimulateDayResponse(
            success=True,
            newDate=target_date.isoformat(),
            changes=Changes(
                itemsUsed=items_used,
                itemsExpired=items_expired,
                itemsDepletedToday=items_depleted_today
            )
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )