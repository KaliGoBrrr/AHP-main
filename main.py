from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from datetime import datetime
import init_db
from bson import ObjectId
import re

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "ahp_database")

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collections
criteria_collection = db["criteria"]
vehicles_collection = db["vehicles"]
logs_collection = db["logs"]
criteria_weights_collection = db["criteria_weights"]  # Collection mới

app = FastAPI()

# Pydantic models
class Criterion(BaseModel):
    name: str

class VehicleDetails(BaseModel):
    brand: str
    year: int
    type: str
    engine: str
    price_range: str

class Vehicle(BaseModel):
    name: str
    image: str
    details: VehicleDetails

class LogData(BaseModel):
    weights: List[float]
    top_result: List[List[Any]]
    criteria_matrices: List[Dict[str, float]]

class CriteriaWeights(BaseModel):
    weights: List[float]

# Initialize database
@app.on_event("startup")
async def startup_event():
    try:
        init_db.initialize_db(db)
        print("Kết nối MongoDB thành công!")
    except Exception as e:
        print(f"Lỗi kết nối MongoDB: {e}")

# API endpoints
@app.get("/get_criteria", response_model=Dict[str, List[Dict[str, Any]]])
async def get_criteria():
    criteria = list(criteria_collection.find({}, {"_id": 0}))
    return {"criteria": criteria}

@app.get("/get_vehicles", response_model=Dict[str, List[Dict[str, Any]]])
async def get_vehicles():
    vehicles = list(vehicles_collection.find({}, {"_id": 0}))
    return {"vehicles": vehicles}

@app.post("/add_criterion", response_model=Dict[str, str])
async def add_criterion(criterion: Criterion):
    existing = criteria_collection.find_one({"name": criterion.name})
    if existing:
        raise HTTPException(status_code=400, detail="Tiêu chí đã tồn tại")
    index = criteria_collection.count_documents({})
    criteria_collection.insert_one({"name": criterion.name, "index": index})
    # Xóa trọng số hiện tại nếu thêm tiêu chí mới
    criteria_weights_collection.delete_one({"_id": "current_weights"})
    return {"message": f"Thêm tiêu chí '{criterion.name}' thành công"}

@app.post("/add_vehicle", response_model=Dict[str, str])
async def add_vehicle(vehicle: Vehicle):
    existing = vehicles_collection.find_one({"name": vehicle.name})
    if existing:
        raise HTTPException(status_code=400, detail="Xe đã tồn tại")
    vehicles_collection.insert_one(vehicle.dict())
    return {"message": f"Thêm xe '{vehicle.name}' thành công"}

@app.delete("/delete_criterion/{name}", response_model=Dict[str, str])
async def delete_criterion(name: str):
    result = criteria_collection.delete_one({"name": name})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Tiêu chí không tồn tại")
    # Update indices
    criteria = list(criteria_collection.find({}, {"_id": 0}))
    for i, crit in enumerate(criteria):
        criteria_collection.update_one({"name": crit["name"]}, {"$set": {"index": i}})
    # Xóa trọng số hiện tại nếu xóa tiêu chí
    criteria_weights_collection.delete_one({"_id": "current_weights"})
    return {"message": f"Xóa tiêu chí '{name}' thành công"}

@app.delete("/delete_vehicle/{name}", response_model=Dict[str, str])
async def delete_vehicle(name: str):
    result = vehicles_collection.delete_one({"name": name})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Xe không tồn tại")
    return {"message": f"Xóa xe '{name}' thành công"}

@app.get("/get_criteria_weights", response_model=Dict[str, List[float]])
async def get_criteria_weights():
    criteria = list(criteria_collection.find({}, {"_id": 0}))
    weights_doc = criteria_weights_collection.find_one({"_id": "current_weights"})
    if weights_doc and "weights" in weights_doc and len(weights_doc["weights"]) == len(criteria):
        return {"weights": weights_doc["weights"]}
    weights = [1.0 / len(criteria)] * len(criteria) if criteria else []
    return {"weights": weights}

@app.post("/save_criteria_weights", response_model=Dict[str, str])
async def save_criteria_weights(weights: CriteriaWeights):
    criteria = list(criteria_collection.find({}, {"_id": 0}))
    if len(weights.weights) != len(criteria):
        raise HTTPException(status_code=400, detail=f"Số trọng số ({len(weights.weights)}) không khớp với số tiêu chí ({len(criteria)})")
    if abs(sum(weights.weights) - 1.0) > 1e-6:
        raise HTTPException(status_code=400, detail="Tổng trọng số phải bằng 1")
    criteria_weights_collection.update_one(
        {"_id": "current_weights"},
        {"$set": {"weights": weights.weights, "timestamp": datetime.utcnow().isoformat()}},
        upsert=True
    )
    return {"message": "Lưu trọng số thành công"}

@app.post("/log_calculation", response_model=Dict[str, str])
async def log_calculation(log_data: LogData):
    log_entry = log_data.dict()
    log_entry["timestamp"] = datetime.utcnow().isoformat()
    result = logs_collection.insert_one(log_entry)
    return {"message": "Lưu log thành công", "log_id": str(result.inserted_id)}

@app.get("/logs", response_model=List[Dict[str, Any]])
async def get_logs():
    logs = list(logs_collection.find())
    for log in logs:
        log["_id"] = str(log["_id"])
    return logs

@app.delete("/logs/{log_id}", response_model=Dict[str, str])
async def delete_log(log_id: str):
    if not re.match(r"^[0-9a-fA-F]{24}$", log_id):
        raise HTTPException(status_code=400, detail="ID log không hợp lệ: Phải là chuỗi 24 ký tự hex.")
    try:
        result = logs_collection.delete_one({"_id": ObjectId(log_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Log với ID {log_id} không tồn tại.")
        return {"message": "Xóa log thành công"}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"ID log {log_id} không phải ObjectId hợp lệ.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi server khi xóa log: {str(e)}")

# Thêm endpoint để kiểm tra trạng thái
@app.get("/check_status", response_model=Dict[str, Any])
async def check_status():
    criteria_count = criteria_collection.count_documents({})
    vehicles_count = vehicles_collection.count_documents({})
    weights_doc = criteria_weights_collection.find_one({"_id": "current_weights"})
    weights_saved = weights_doc is not None and "weights" in weights_doc and len(weights_doc["weights"]) == criteria_count
    return {
        "criteria_count": criteria_count,
        "vehicles_count": vehicles_count,
        "weights_saved": weights_saved
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)