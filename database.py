from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
import os
from dotenv import load_dotenv
from init_db import init_criteria, init_vehicles

# Tải biến môi trường
load_dotenv()

def connect_to_mongo():
    """Kết nối tới MongoDB với xử lý lỗi."""
    try:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        client = MongoClient(mongo_uri)
        client.admin.command("ping")
        print("Kết nối MongoDB thành công!")
        return client
    except ConnectionFailure as e:
        print(f"Lỗi kết nối MongoDB: {e}")
        raise
    except Exception as e:
        print(f"Lỗi không xác định khi kết nối MongoDB: {e}")
        raise

def initialize_db(criteria_collection, vehicles_collection, force=False):
    """Khởi tạo cơ sở dữ liệu bằng cách khởi tạo tiêu chí và xe."""
    try:
        # Khởi tạo tiêu chí
        init_criteria(criteria_collection, force)
        # Khởi tạo xe
        init_vehicles(vehicles_collection, force)
        print("Khởi tạo cơ sở dữ liệu thành công!")
    except PyMongoError as e:
        print(f"Lỗi MongoDB khi khởi tạo cơ sở dữ liệu: {str(e)}")
        raise
    except Exception as e:
        print(f"Lỗi không xác định khi khởi tạo cơ sở dữ liệu: {str(e)}")
        raise

# Kết nối tới MongoDB
client = connect_to_mongo()
db = client["ahp_database"]
vehicles_collection = db["vehicles"]
log_collection = db["logs"]
criteria_collection = db["criteria"]

# Cleanup khi ứng dụng dừng
import atexit
atexit.register(client.close)

if __name__ == "__main__":
    print("Cảnh báo: Chạy file này sẽ xóa và khởi tạo lại dữ liệu!")
    initialize_db(criteria_collection, vehicles_collection, force=True)