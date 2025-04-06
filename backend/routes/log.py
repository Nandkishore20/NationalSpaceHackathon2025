from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import sqlite3

router = APIRouter()

class LogDetail(BaseModel):
    fromContainer: Optional[str] = None
    toContainer: Optional[str] = None
    reason: Optional[str] = None

class LogEntry(BaseModel):
    timestamp: str
    userId: str
    actionType: str
    itemId: str
    details: LogDetail

class LogResponse(BaseModel):
    logs: List[LogEntry]

@router.get("/logs", response_model=LogResponse)
async def get_logs(
    startDate: str = Query(..., description="Start date in ISO format (YYYY-MM-DD)"),
    endDate: str = Query(..., description="End date in ISO format (YYYY-MM-DD)")
):
    """
    Retrieve logs within a specified date range.
    
    Args:
        startDate: ISO formatted start date string
        endDate: ISO formatted end date string
        
    Returns:
        LogResponse: List of log entries within the specified date range
    """
    try:
        # Validate date formats
        try:
            # Replace 'Z' with an empty string if it exists
            start_date_clean = startDate.replace('Z', '') if 'Z' in startDate else startDate
            end_date_clean = endDate.replace('Z', '') if 'Z' in endDate else endDate
            
            datetime.fromisoformat(start_date_clean)
            datetime.fromisoformat(end_date_clean)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format (YYYY-MM-DDThh:mm:ss)")
        
        # Connect to database
        conn = sqlite3.connect('inventory.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Query logs within the date range
        cursor.execute("""
            SELECT log_id, item_id, user_id, timestamp, container_id, from_container, to_container, reason
            FROM usage_log
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        """, (start_date_clean, end_date_clean))
        
        log_rows = cursor.fetchall()
        conn.close()
        
        # Format results according to response model
        logs = []
        for row in log_rows:
            # Determine action type based on available data
            action_type = "unknown"
            if row["reason"] == "retrieval":
                action_type = "retrieve"
            elif row["reason"] == "rearrangement":
                action_type = "move"
            
            # Create log entry
            log_entry = LogEntry(
                timestamp=row["timestamp"],
                userId=row["user_id"],
                actionType=action_type,
                itemId=row["item_id"],
                details=LogDetail(
                    fromContainer=row["from_container"],
                    toContainer=row["to_container"],
                    reason=row["reason"]
                )
            )
            logs.append(log_entry)
        
        return LogResponse(logs=logs)
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error retrieving logs: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)