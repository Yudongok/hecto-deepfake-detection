import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def analyze_frequency(image_path):
    # 1. 이미지 불러오기 & 전처리
    try:
        # 이미지를 흑백(Grayscale)으로 엽니다. 
        # (주파수 분석은 보통 색상보다 밝기 패턴에서 흔적을 찾기 때문입니다)
        img = Image.open(image_path).convert('L') 
    except FileNotFoundError:
        print(f"❌ 오류: '{image_path}' 파일을 찾을 수 없습니다. 폴더에 이미지를 넣어주세요!")
        return

    # 이미지를 224x224로 리사이즈하고 Tensor로 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img) # Shape: [1, 224, 224] (C, H, W)
    
    # 2. FFT (Fast Fourier Transform) 수행
    # torch.fft.fft2: 2차원 이미지를 주파수 도메인으로 변환
    fft_result = torch.fft.fft2(img_tensor)
    
    # 3. 주파수 중심 이동 (Shift)
    # 원래 FFT 결과는 모서리가 저주파인데, 보기 좋게 저주파를 중앙으로 모읍니다.
    fft_shifted = torch.fft.fftshift(fft_result)
    
    # 4. 시각화를 위한 절대값(Magnitude) 및 로그 스케일 변환
    # 주파수 값의 차이가 너무 커서 로그(log)를 취해야 눈에 보입니다.
    magnitude = torch.abs(fft_shifted)
    log_magnitude = 20 * torch.log(magnitude + 1e-6) # 0으로 나누기 방지
    
    # Tensor를 Numpy로 변환 (시각화용)
    original_img_np = np.array(img.resize((224, 224)))
    fft_img_np = log_magnitude.squeeze().numpy()

    # 5. 결과 시각화
    plt.figure(figsize=(12, 6))

    # 왼쪽: 원본 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(original_img_np, cmap='gray')
    plt.title("Original Image (Spatial Domain)")
    plt.axis('off')

    # 오른쪽: 주파수 스펙트럼
    plt.subplot(1, 2, 2)
    plt.imshow(fft_img_np, cmap='inferno') # 열화상 카메라 느낌의 색상
    plt.title("Frequency Spectrum (Log Magnitude)")
    plt.colorbar(label='Magnitude (dB)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    print("✅ 시각화 완료! 창이 떴는지 확인해보세요.")

if __name__ == "__main__":
    # 여기에 준비한 이미지 파일 이름을 넣으세요.
    analyze_frequency("test_image.jpg")