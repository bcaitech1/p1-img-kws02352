import segmentation_models_pytorch as smp

# model 불러오기
# 출력 레이블 수 정의 (classes = 12)
model = smp.DeepLabV3Plus(encoder_name='resnext101_32x16d', classes=12 , encoder_weights='swsl', activation=None)
model = model.to(device)