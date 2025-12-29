import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os

def display_dicom_with_matplotlib(dicom_file_path):
    """
    DICOM 파일을 로드하고 matplotlib을 사용하여 영상을 표시하며 태그 정보를 출력합니다.

    Args:
        dicom_file_path (str): 로드할 DICOM 파일의 경로.
    """
    try:
        if not os.path.exists(dicom_file_path):
            print(f"오류: 파일을 찾을 수 없습니다 - {dicom_file_path}")
            print("DICOM 파일을 Colab에 업로드하거나 올바른 경로를 지정해 주세요.")
            return

        # pydicom으로 DICOM 파일 읽기
        dicom_data = pydicom.dcmread(dicom_file_path)
        pixel_array = dicom_data.pixel_array  # 픽셀 데이터 추출 (2D 배열)

        # 영상 크기 추출 (Rows와 Columns 태그)
        rows = dicom_data.get((0x0028, 0x0010), None)
        cols = dicom_data.get((0x0028, 0x0011), None)
        print(f"Image Size: {rows.value if rows else 'N/A'} x {cols.value if cols else 'N/A'}")

        # 픽셀 데이터를 0-255 범위로 정규화 (uint8 타입으로 변환)
        pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
        pixel_array = pixel_array.astype(np.uint8)

        # Matplotlib로 이미지 생성 및 표시
        plt.figure(figsize=(6, 6)) # 이미지 표시 크기 설정
        plt.imshow(pixel_array, cmap='gray')  # 흑백 컬러맵 사용
        plt.title(f"Loaded: {os.path.basename(dicom_file_path)}")
        plt.axis('off')  # 축 숨김
        plt.show()

        # DICOM 태그 정보 출력
        print("\n--- DICOM 태그 정보 ---")
        for elem in dicom_data:
            try:
                print(f"{elem.tag} | {elem.name:<40} | {str(elem.value)}")
            except Exception as e:
                print(f"{elem.tag} | {'Error decoding':<40} | {str(e)}")

    except Exception as e:
        print(f"오류 발생: {str(e)}")


# --- 사용 예시 ---
# 여기에 DICOM 파일의 경로를 입력해 주세요.
# 예시: dicom_file = '/content/your_dicom_file.dcm'
# Colab 환경에서는 파일을 직접 업로드하거나 Google Drive에서 로드해야 합니다.

# TODO: 사용자에게 DICOM 파일을 업로드하라고 안내하는 것이 더 좋을 수 있습니다.
# 현재는 예시 경로를 사용합니다.

# 파일을 업로드할 준비가 되면, 다음 줄의 주석을 해제하고 파일 이름을 지정하세요.
# from google.colab import files
# uploaded = files.upload()
# dicom_file = list(uploaded.keys())[0] if uploaded else None

dicom_file_path = '/content/medical_image.dcm' # 예시 파일 경로, 실제 파일 경로로 변경 필요

# Colab 환경에서 파일을 업로드하려면 다음 코드를 사용하세요:
# from google.colab import files
# uploaded = files.upload()
# if uploaded:
#     # 첫 번째 업로드된 파일 이름을 가져옵니다.
#     dicom_file_path = list(uploaded.keys())[0]
#     display_dicom_with_matplotlib(dicom_file_path)
# else:
#     print("파일 업로드가 취소되었습니다.")

# 임시로 더미 파일을 만들어 테스트하거나, 실제 DICOM 파일을 업로드하여 사용하세요.
# 실제 DICOM 파일을 가지고 있다면, 아래 `dicom_file_path`를 해당 파일의 경로로 수정하세요.

print("DICOM 파일을 Colab 환경에 업로드하거나, Google Drive에 있는 DICOM 파일의 경로를 지정해주세요.")
print("예시: `files.upload()`를 사용하여 업로드하거나, `dicom_file_path = '/content/my_dicom_scan.dcm'` 등으로 경로를 지정할 수 있습니다.")
print("아래 코드는 현재 `dicom_file_path` 변수에 지정된 경로를 사용하려고 시도합니다.")

# 파일을 업로드 한 후 다음 함수를 호출하세요.
# display_dicom_with_matplotlib(dicom_file_path)