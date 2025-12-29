import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("C:\img\dcm2_ori_blur.bmp", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not read image file.")
    exit()

# 가우시안 블러를 적용하여 저주파 성분 추출 (Low Pass Filter 효과)
# (커널 크기는 홀수여야 합니다. 예: (21, 21))
blurred_img = cv2.GaussianBlur(img, (21, 21), 0)

# 원본 이미지에서 블러 이미지를 빼서 고주파 성분만 추출
# 결과값의 범위를 조정하기 위해 127을 더해 회색조로 표현하기도 합니다.
g_hpf_img = img - blurred_img + 127 #

# 결과 시각화 및 저장 (위의 코드 참고)
plt.imshow(g_hpf_img, cmap='gray')
plt.title('Gaussian HPF Image')
plt.show()