import torch
import torch.nn as nn
import timm

def inspect_and_modify_model():
    # 1. 모델 로드 (EfficientNet-B0 사용)
    # pretrained=True: 이미 ImageNet 데이터로 학습된 똑똑한 가중치를 가져옵니다.
    # num_classes=0: Head(분류기)를 제거하고 Backbone만 가져오는 옵션도 있지만,
    # 구조를 보기 위해 일단 1000개 클래스(기본값)로 로드합니다.
    print("=== [1] 모델 로드 중... ===")
    model_name = 'efficientnet_b0'
    model = timm.create_model(model_name, pretrained=True)
    
    # 2. 모델의 전체 구조 확인 (너무 길어서 주석 처리, 필요하면 주석 해제)
    # print(model) 

    # 3. Backbone의 출력 채널 수 확인 (Head에 들어갈 입력 크기)
    # EfficientNet이나 ResNet 등 모델마다 마지막 Feature Map의 채널 수가 다릅니다.
    # timm은 이걸 'num_features'로 편하게 제공합니다.
    n_features = model.num_features
    print(f"✅ 모델명: {model_name}")
    print(f"✅ Backbone이 뱉어내는 특징(Feature) 개수: {n_features}")

    # 4. 현재 붙어있는 Head(분류기) 확인
    # timm 모델들은 보통 'classifier' 또는 'fc'라는 이름으로 Head를 가집니다.
    print(f"\n=== [2] 현재 Head 구조 (변경 전) ===")
    print(model.classifier) 

    # ---------------------------------------------------------
    # [핵심] 대회 전략: Custom Head로 교체하기
    # ---------------------------------------------------------
    
    # 5. 기존 Head를 내 입맛대로 교체 (Binary Classification: Real vs Fake)
    # 단순 Linear 하나가 아니라, 좀 더 두꺼운 층을 쌓을 수도 있습니다.
    my_custom_head = nn.Sequential(
        nn.LayerNorm(n_features),      # 정규화 (학습 안정성)
        nn.Dropout(0.3),               # 과적합 방지
        nn.Linear(n_features, 1)       # 최종 출력: 1개 (0~1 사이의 확률값)
    )
    
    model.classifier = my_custom_head
    
    print(f"\n=== [3] 교체된 Custom Head 구조 (변경 후) ===")
    print(model.classifier)
    
    # 6. 데이터 통과시켜보기 (Forward Pass 테스트)
    # 가짜 이미지 데이터 생성 (Batch: 2, Channel: 3, Height: 224, Width: 224)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # 모델 추론
    output = model(dummy_input)
    
    print(f"\n=== [4] 추론 테스트 ===")
    print(f"입력 크기: {dummy_input.shape}")
    print(f"출력 크기: {output.shape} (예상: [2, 1])")
    print("🎉 성공적으로 모델 구조를 변경하고 실행했습니다!")

if __name__ == "__main__":
    inspect_and_modify_model()