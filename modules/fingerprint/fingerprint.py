import zipfile
import os
import cv2
import numpy as np
import sys
zip_file_name = 'van tay.zip'

if os.path.exists(zip_file_name):
    print(f"Đang giải nén file {zip_file_name}...")

    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall('.') 

    print(f"Đã giải nén xong! Bạn sẽ thấy thư mục 'van tay'.")
    
else:
    print(f"[LỖI] Không tìm thấy file {zip_file_name}. Bạn đã upload nó chưa?")

# --- 1. CÀI ĐẶT ---
DATASET_PATH = 'van tay' 
IMAGE_TO_CHECK = 'QD1.tif'

MIN_MATCH_COUNT = 15 # Giữ nguyên 15, bạn có thể tăng lên 20-25 nếu muốn

# --- HÀM MỚI: DÙNG ĐỂ LÀM RÕ ẢNH VÂN TAY ---
def process_image(img_gray):
    """
    Áp dụng Adaptive Thresholding để làm rõ vân tay.
    """
    if img_gray is None:
        return None
        
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    img_binary = cv2.adaptiveThreshold(
        img_blur, 
        255, # Giá trị tối đa (màu trắng)
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Phương pháp
        cv2.THRESH_BINARY_INV, # Đảo ngược: vân tay = TRẮNG, nền = ĐEN
        11, # Kích thước vùng lân cận (phải là số lẻ)
        2   # Hằng số C (trừ đi từ giá trị trung bình)
    )
    
    return img_binary
# --- KẾT THÚC HÀM MỚI ---


# --- 2. HÀM "HUẤN LUYỆN" (TẢI CSDL) - ĐÃ SỬA ---
def load_dataset_descriptors(path):
    """
    Tải tất cả ảnh, XỬ LÝ ẢNH, tính ORB và lưu vào list.
    """
    print(f"[INFO] Đang tải CSDL vân tay từ: {path}")
    known_descriptors = []
    
    orb = cv2.ORB_create(nfeatures=1000)
    
    if not os.path.exists(path):
        print(f"[LỖI] Không tìm thấy thư mục CSDL: {path}")
        sys.exit()

    for filename in os.listdir(path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp','.tif','.tiff')):
            continue # Bỏ qua các file không phải ảnh
            
        try:
            name = os.path.splitext(filename)[0]
            
            img_path = os.path.join(path, filename)
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img_gray is None:
                print(f"[CẢNH BÁO] Bỏ qua file: {filename} (không thể đọc)")
                continue
                
            img_processed = process_image(img_gray)
            
            if img_processed is None:
                continue
                
            kp, des = orb.detectAndCompute(img_processed, None)
            
            if des is None:
                print(f"[CẢNH BÁO] Bỏ qua file: {filename} (không tìm thấy đặc điểm sau khi xử lý)")
                continue
                
            known_descriptors.append((name, kp, des, img_gray)) 
            print(f"  -> Đã xử lý: {name} (tìm thấy {len(kp)} điểm)")
            
        except Exception as e:
            print(f"[LỖI] Xử lý file {filename} thất bại: {e}")
            
    print(f"[INFO] Đã tải xong CSDL ({len(known_descriptors)} người dùng).")
    return known_descriptors

# --- 3. HÀM SO SÁNH (TÌM KHỚP NHẤT) - ĐÃ SỬA ---
def find_best_match(image_to_check_path, known_descriptors, min_threshold):
    """
    So sánh 1 ảnh (đã qua xử lý) với TẤT CẢ CSDL đã tải.
    """
    
    # --- A. Tải và xử lý ảnh cần check ---
    if not os.path.isfile(image_to_check_path):
        print(f"[LỖI] Không tìm thấy ảnh cần check: {image_to_check_path}")
        return
        
    img_to_check_gray = cv2.imread(image_to_check_path, cv2.IMREAD_GRAYSCALE)
    if img_to_check_gray is None:
        print(f"[LỖI] Không thể đọc ảnh cần check.")
        return

    print("[INFO] Đang xử lý ảnh cần check (ảnh mờ)...")
    img_to_check_processed = process_image(img_to_check_gray)
    
    if img_to_check_processed is None:
        print("[LỖI] Xử lý ảnh cần check thất bại.")
        return

    orb = cv2.ORB_create(nfeatures=1000)
    kp_new, des_new = orb.detectAndCompute(img_to_check_processed, None)
    
    if des_new is None:
        print("[LỖI] Không tìm thấy đặc điểm nào trên ảnh cần check (kể cả sau khi xử lý).")
        return

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # --- B. Vòng lặp so sánh với CSDL ---
    print("\n[INFO] Đang so sánh ảnh đã xử lý với CSDL...")
    
    best_score = 0
    best_name = "Khong ro"
    best_match_img = None
    best_kp = None
    best_good_matches = []

    for (name, kp_known, des_known, img_known_original) in known_descriptors:
        try:
            matches = bf.knnMatch(des_known, des_new, k=2) # So sánh des (đã xử lý)
            good_matches = []
            for m, n in matches:
                if n.distance > 0 and m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            num_good_matches = len(good_matches)
            print(f"  -> So với '{name}': tìm thấy {num_good_matches} điểm khớp.")
            
            if num_good_matches > best_score:
                best_score = num_good_matches
                best_name = name
                best_match_img = img_known_original # Lưu ảnh GỐC để vẽ
                best_kp = kp_known
                best_good_matches = good_matches
                
        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi khi so sánh với {name}: {e}")
            continue

    # --- C. In kết quả cuối cùng ---
    print(f"\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(f"Tìm thấy khớp nhất là: {best_name}")
    print(f"Số điểm khớp cao nhất: {best_score}")

    if best_score >= min_threshold:
        print(f"KẾT LUẬN: GIỐNG NHAU (Vì {best_score} >= {min_threshold})")
        
        img_result = cv2.drawMatches(
            best_match_img, best_kp, 
            img_to_check_gray, kp_new, # Dùng ảnh gốc và kp_new
            best_good_matches, None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        h, w = img_result.shape[:2]
        if h > 800:
            scale = 800 / h
            img_result = cv2.resize(img_result, (int(w * scale), int(h * scale)))
            
        cv2.imshow(f"Ket qua: Khop voi {best_name}", img_result)
        
    else:
        print(f"KẾT LUẬN: KHÁC NHAU (Vì {best_score} < {min_threshold})")
        cv2.imshow("Ket qua: Khong khop", img_to_check_gray)

    print("\n[INFO] Nhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- 4. HÀM CHÍNH ĐỂ CHẠY ---
def main():
    known_data = load_dataset_descriptors(DATASET_PATH)
    if not known_data:
        print("[LỖI] CSDL rỗng. Dừng chương trình.")
        return
    find_best_match(IMAGE_TO_CHECK, known_data, MIN_MATCH_COUNT)

if __name__ == "__main__":
    main()