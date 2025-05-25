from pymongo import MongoClient
from pymongo.errors import PyMongoError
import os
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv()

def init_criteria(criteria_collection, force=False):
    """Khởi tạo danh sách tiêu chí mặc định."""
    try:
        if force or criteria_collection.count_documents({}) == 0:
            default_criteria = [
                {"name": "Độ bền", "index": 0},
                {"name": "Hiệu suất", "index": 1},
                {"name": "Thiết kế", "index": 2},
                {"name": "Đánh giá", "index": 3},
                {"name": "Giá bán", "index": 4}
            ]
            # Kiểm tra tính duy nhất của name
            names = [c["name"] for c in default_criteria]
            if len(names) != len(set(names)):
                raise ValueError("Tiêu chí mặc định chứa tên trùng lặp.")
            if force:
                criteria_collection.drop()
            criteria_collection.insert_many(default_criteria)
            print("Khởi tạo tiêu chí thành công!")
        else:
            print("Tiêu chí đã tồn tại, bỏ qua khởi tạo.")
    except PyMongoError as e:
        print(f"Lỗi MongoDB khi khởi tạo tiêu chí: {str(e)}")
        raise
    except Exception as e:
        print(f"Lỗi không xác định khi khởi tạo tiêu chí: {str(e)}")
        raise

def init_vehicles(vehicles_collection, force=False):
    """Khởi tạo danh sách xe mặc định."""
    try:
        if force or vehicles_collection.count_documents({}) == 0:
            default_vehicles = [
                {
                    "name": "Toyota Camry",
                    "image": "toyota_camry.jpg",
                    "details": {
                        "brand": "Toyota",
                        "year": 2023,
                        "type": "Sedan",
                        "engine": "2.5L 4-cylinder",
                        "price_range": "800-900 triệu VNĐ"
                    }
                },
                {
                    "name": "Honda Civic",
                    "image": "honda_civic.jpg",
                    "details": {
                        "brand": "Honda",
                        "year": 2023,
                        "type": "Sedan",
                        "engine": "1.5L Turbo",
                        "price_range": "730-870 triệu VNĐ"
                    }
                },
                {
                    "name": "Mazda CX-5",
                    "image": "mazda_cx5.jpg",
                    "details": {
                        "brand": "Mazda",
                        "year": 2023,
                        "type": "SUV",
                        "engine": "2.0L Skyactiv-G",
                        "price_range": "750-900 triệu VNĐ"
                    }
                },
                {
                    "name": "Ford Ranger",
                    "image": "ford_ranger.jpg",
                    "details": {
                        "brand": "Ford",
                        "year": 2023,
                        "type": "Bán tải",
                        "engine": "2.0L Turbo",
                        "price_range": "650-1.2 tỷ VNĐ"
                    }
                },
                {
                    "name": "Hyundai Tucson",
                    "image": "hyundai_tucson.jpg",
                    "details": {
                        "brand": "Hyundai",
                        "year": 2023,
                        "type": "SUV",
                        "engine": "2.0L",
                        "price_range": "825-1 tỷ VNĐ"
                    }
                },
                {
                    "name": "Kia Seltos",
                    "image": "kia_seltos.jpg",
                    "details": {
                        "brand": "Kia",
                        "year": 2023,
                        "type": "SUV cỡ B",
                        "engine": "1.5L",
                        "price_range": "600-720 triệu VNĐ"
                    }
                },
                {
                    "name": "Mitsubishi Xpander",
                    "image": "mitsubishi_xpander.jpg",
                    "details": {
                        "brand": "Mitsubishi",
                        "year": 2023,
                        "type": "MPV",
                        "engine": "1.5L",
                        "price_range": "560-670 triệu VNĐ"
                    }
                },
                {
                    "name": "VinFast VF8",
                    "image": "vinfast_vf8.jpg",
                    "details": {
                        "brand": "VinFast",
                        "year": 2023,
                        "type": "SUV điện",
                        "engine": "Electric Dual Motor",
                        "price_range": "1.1-1.3 tỷ VNĐ"
                    }
                }
            ]
            # Kiểm tra tính duy nhất của name
            names = [v["name"] for v in default_vehicles]
            if len(names) != len(set(names)):
                raise ValueError("Xe mặc định chứa tên trùng lặp.")
            if force:
                vehicles_collection.drop()
            result = vehicles_collection.insert_many(default_vehicles)
            print(f"Khởi tạo xe thành công! Đã thêm {len(default_vehicles)} xe với ID: {result.inserted_ids}")
        else:
            print("Xe đã tồn tại, bỏ qua khởi tạo.")
    except PyMongoError as e:
        print(f"Lỗi MongoDB khi khởi tạo xe: {str(e)}")
        raise
    except Exception as e:
        print(f"Lỗi không xác định khi khởi tạo xe: {str(e)}")
        raise

def initialize_db(db, force=False):
    """Khởi tạo cơ sở dữ liệu với các collection criteria, vehicles, logs."""
    try:
        criteria_collection = db["criteria"]
        vehicles_collection = db["vehicles"]
        logs_collection = db["logs"]

        # Khởi tạo criteria
        init_criteria(criteria_collection, force=force)

        # Khởi tạo vehicles
        init_vehicles(vehicles_collection, force=force)

        # Đảm bảo collection logs tồn tại
        if logs_collection.count_documents({}) == 0:
            print("Collection logs đã sẵn sàng.")
        else:
            if force:
                logs_collection.drop()
                print("Đã xóa và tạo lại collection logs.")
            else:
                print("Collection logs đã tồn tại, bỏ qua khởi tạo.")

        print("Khởi tạo cơ sở dữ liệu thành công!")
    except Exception as e:
        print(f"Lỗi khi khởi tạo cơ sở dữ liệu: {str(e)}")
        raise

if __name__ == "__main__":
    print("Cảnh báo: Chạy file này sẽ xóa và khởi tạo lại dữ liệu!")
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client["ahp_database"]
    initialize_db(db, force=True)