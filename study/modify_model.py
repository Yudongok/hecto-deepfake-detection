import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# ---------------------------------------------------------
# [Part 1] 모델 개조 함수 (3채널 -> 4채널)
# ---------------------------------------------------------
def get_4channel_model(model_name='efficientnet_b0', num_classes=1):
    print(f"🔧 모델 개조 시작: {model_name} (3ch -> 4ch)")
    
    # 1. 기존 Pretrained 모델 로드
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
    # 2. 첫 번째 레이어(Conv2d) 찾기
    # EfficientNet은 보통 'conv_stem'이라는 이름으로 첫 레이어를 가집니다.
    # 모델마다 이름이 다를 수 있으니 확인이 필요하지만, EfficientNet 계열은 이게 맞습니다.
    original_layer = model.conv_stem
    
    # 3. 새로운 4채널 레이어 생성
    # 기존 설정(out_channels, kernel_size, stride 등)은 그대로 유지하고 in_channels만 4로 변경
    new_layer = nn.Conv2d(
        in_channels=4, 
        out_channels=original_layer.out_channels,
        kernel_size=original_layer.kernel_size,
        stride=original_layer.stride,
        padding=original_layer.padding,
        bias=original_layer.bias
    )
    
    # 4. [중요] 가중치(Weight) 이식 (Transfer Learning)
    # 기존 3채널(RGB)의 지식은 그대로 복사해옵니다.
    with torch.no_grad():
        # (1) RGB 채널 복사: 기존 가중치를 그대로 넣음
        new_layer.weight[:, :3, :, :] = original_layer.weight
        
        # (2) 주파수 채널 초기화: 
        # 그냥 0으로 두거나 랜덤으로 두면 학습 초반에 튈 수 있습니다.
        # RGB 채널의 평균값으로 초기화하여 모델이 놀라지 않게 합니다.
        new_layer.weight[:, 3:4, :, :] = torch.mean(original_layer.weight, dim=1, keepdim=True)
        
    # 5. 모델의 첫 레이어를 교체
    model.conv_stem = new_layer
    
    print("✅ 모델 개조 완료! 첫 레이어가 4채널 입력을 받습니다.")
    return model

# ---------------------------------------------------------
# [Part 2] 데이터 전처리 함수 (이미지 -> 4채널 텐서)
# ---------------------------------------------------------
def process_image_4ch(image_path):
    # 1. 이미지 로드
    try:
        img_rgb = Image.open(image_path).convert('RGB')
    except:
        # 테스트를 위해 이미지가 없으면 랜덤 생성
        img_rgb = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # 2. 기본 Transform (Resize & Tensor 변환)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # 값 범위: 0.0 ~ 1.0
    ])
    
    rgb_tensor = transform(img_rgb) # Shape: [3, 224, 224]
    
    # 3. FFT 채널 생성 (아까 본 그 과정을 압축)
    gray_tensor = transforms.Grayscale()(rgb_tensor) # 밝기 정보만 사용
    fft = torch.fft.fft2(gray_tensor)
    fft_shifted = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shifted)
    fft_log = torch.log(magnitude + 1e-6) # 로그 스케일
    
    # [중요] 정규화 (Normalization)
    # FFT 값은 수십~수백까지 올라갑니다. RGB(0~1)와 맞추기 위해 0~1 사이로 눌러줍니다.
    fft_norm = (fft_log - fft_log.min()) / (fft_log.max() - fft_log.min())
    
    # 4. 합치기 (Concatenate)
    # RGB(3) + FFT(1) = 4채널
    input_tensor = torch.cat([rgb_tensor, fft_norm], dim=0) # Shape: [4, 224, 224]
    
    return input_tensor.unsqueeze(0) # Batch 차원 추가 -> [1, 4, 224, 224]

# ---------------------------------------------------------
# [Part 3] 실행 테스트
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. 모델 준비
    model = get_4channel_model('efficientnet_b0')
    
    # 2. 데이터 준비 (폴더에 test_image.jpg가 있으면 그걸 쓰고, 없으면 랜덤)
    input_data = process_image_4ch("test_image.jpg")
    
    print(f"\n📊 입력 데이터 형태: {input_data.shape} (Batch, Channel, Height, Width)")
    
    # 3. 추론
    output = model(input_data)
    print(f"🚀 출력 결과: {output.shape} (Real/Fake 점수)")
    print("🎉 4채널 파이프라인이 정상 작동합니다!")
    
    # ... (기존 코드 뒤에 추가) ...
    
    # 1. 껍데기 벗겨서 실제 값 보기
    raw_score = output.item() 
    print(f"\n🔢 모델의 날것 점수 (Logit): {raw_score:.4f}")

    # 2. 확률로 변환하기 (Sigmoid 함수: 0~1 사이 값으로 압축)
    probability = torch.sigmoid(output).item()
    print(f"📊 가짜(Fake)일 확률: {probability * 100:.2f}%")

    # 3. 최종 판결 (기준점 0.5)
    if probability > 0.5:
        print("🚨 판결: [Fake / 딥페이크] 입니다!")
    else:
        print("✅ 판결: [Real / 진짜] 입니다!")