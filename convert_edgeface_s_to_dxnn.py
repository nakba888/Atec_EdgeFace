"""
EdgeFace s_gamma_05 모델을 DXNN으로 변환하는 스크립트

변환 단계:
1. PyTorch (.pt) -> ONNX (.onnx)
2. Calibration dataset 준비
3. Calibration config 생성
4. ONNX -> DXNN (DeepX 컴파일러 사용)
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# EdgeFace 백본 모델 임포트
sys.path.insert(0, 'face_alignment')
from backbones import get_model


# ============================================================================
# 설정
# ============================================================================

# 모델 경로
PYTORCH_MODEL = "checkpoints/edgeface_s_gamma_05.pt"
ONNX_OUTPUT = "checkpoints/edgeface_s_gamma_05.onnx"
DXNN_OUTPUT = "checkpoints/edgeface_s_gamma_05.dxnn"

# 모델 이름
MODEL_NAME = "edgeface_s_gamma_05"

# Calibration 설정
CALIBRATION_DATASET = "npu_calibration/calibration_dataset"
CALIBRATION_CONFIG = "npu_calibration/calibration_config_edgeface_s.json"
NUM_CALIBRATION_SAMPLES = 100

# LFW 경로 (calibration dataset 생성용)
LFW_DIR = "/mnt/c/Users/Admin/Downloads/lfw-deepfunneled/lfw-deepfunneled"


# ============================================================================
# Step 1: PyTorch -> ONNX 변환
# ============================================================================

def convert_pytorch_to_onnx(
    pytorch_model_path,
    onnx_output_path,
    model_name,
    input_size=(1, 3, 112, 112),
    opset_version=11
):
    """
    PyTorch 모델을 ONNX로 변환

    Args:
        pytorch_model_path: PyTorch 모델 경로 (.pt)
        onnx_output_path: ONNX 출력 경로 (.onnx)
        model_name: 모델 아키텍처 이름
        input_size: 입력 텐서 크기 (batch, channels, height, width)
        opset_version: ONNX opset 버전

    Returns:
        True if successful
    """
    print("=" * 80)
    print("Step 1: PyTorch -> ONNX 변환")
    print("=" * 80)

    try:
        # 디바이스 설정
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # 모델 로드
        print(f"\nLoading PyTorch model: {pytorch_model_path}")
        model = get_model(model_name, fp16=False)
        model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
        model.to(device)
        model.eval()
        print("PyTorch model loaded successfully")

        # 더미 입력 생성
        dummy_input = torch.randn(*input_size).to(device)
        print(f"Dummy input shape: {dummy_input.shape}")

        # ONNX 변환
        print(f"\nExporting to ONNX: {onnx_output_path}")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input.1'],
            output_names=['output'],
            dynamic_axes={
                'input.1': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"\n✅ ONNX export successful: {onnx_output_path}")

        # ONNX 모델 검증
        print("\nVerifying ONNX model...")
        import onnx
        onnx_model = onnx.load(onnx_output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")

        return True

    except Exception as e:
        print(f"\n❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Step 2: Calibration Dataset 준비
# ============================================================================

def prepare_calibration_dataset(lfw_dir, output_dir, num_samples=100):
    """
    Calibration dataset 준비

    Args:
        lfw_dir: LFW 데이터셋 경로
        output_dir: Calibration dataset 출력 경로
        num_samples: Calibration 샘플 수

    Returns:
        True if successful
    """
    print("\n" + "=" * 80)
    print("Step 2: Calibration Dataset 준비")
    print("=" * 80)

    try:
        from npu_calibration.prepare_calibration_dataset import prepare_calibration_dataset as prepare_dataset

        stats = prepare_dataset(
            source_dir=lfw_dir,
            output_dir=output_dir,
            num_samples=num_samples,
            target_size=(112, 112),  # EdgeFace 입력 크기
            quality_threshold=40,
            save_analysis=True
        )

        print(f"\n✅ Calibration dataset prepared: {output_dir}")
        print(f"   Images selected: {stats['images_selected']}")

        return True

    except Exception as e:
        print(f"\n❌ Calibration dataset preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Step 3: Calibration Config 생성
# ============================================================================

def generate_calibration_config(dataset_path, config_output_path, num_samples=100):
    """
    Calibration configuration 생성

    Args:
        dataset_path: Calibration dataset 경로
        config_output_path: Config 출력 경로
        num_samples: Calibration 샘플 수

    Returns:
        True if successful
    """
    print("\n" + "=" * 80)
    print("Step 3: Calibration Config 생성")
    print("=" * 80)

    try:
        from npu_calibration.generate_calibration_config import generate_edgeface_calibration_config

        config = generate_edgeface_calibration_config(
            dataset_path=dataset_path,
            output_path=config_output_path,
            input_name="input.1",
            input_shape=(1, 3, 112, 112),
            calibration_num=num_samples,
            calibration_method="ema",
            use_arcface_preprocessing=True
        )

        print(f"\n✅ Calibration config generated: {config_output_path}")

        return True

    except Exception as e:
        print(f"\n❌ Calibration config generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Step 4: ONNX -> DXNN 변환 (수동)
# ============================================================================

def print_dxnn_conversion_instructions(onnx_path, config_path, dxnn_output):
    """
    ONNX -> DXNN 변환 명령어 안내

    Args:
        onnx_path: ONNX 모델 경로
        config_path: Calibration config 경로
        dxnn_output: DXNN 출력 경로
    """
    print("\n" + "=" * 80)
    print("Step 4: ONNX -> DXNN 변환")
    print("=" * 80)

    print("\n⚠️  이 단계는 DeepX NPU 컴파일러가 필요합니다.")
    print("\n다음 명령어를 실행하여 ONNX를 DXNN으로 변환하세요:\n")

    print("=" * 80)
    print("DeepX Compiler 명령어:")
    print("=" * 80)
    print(f"""
dx_compiler \\
    --model {onnx_path} \\
    --config {config_path} \\
    --output {dxnn_output} \\
    --target npu \\
    --optimize
""")
    print("=" * 80)

    print("\n또는 DeepX 문서의 지침에 따라 변환하세요.")
    print("\n변환 완료 후:")
    print(f"  - DXNN 모델: {dxnn_output}")
    print(f"  - EdgeFace NPU 인식기에서 사용 가능")


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 실행 함수"""

    print("\n" + "=" * 80)
    print("EdgeFace s_gamma_05 -> DXNN 변환 스크립트")
    print("=" * 80)

    print(f"\nPyTorch 모델: {PYTORCH_MODEL}")
    print(f"ONNX 출력: {ONNX_OUTPUT}")
    print(f"DXNN 출력: {DXNN_OUTPUT}")
    print(f"모델 이름: {MODEL_NAME}")

    # Step 1: PyTorch -> ONNX
    success = convert_pytorch_to_onnx(
        pytorch_model_path=PYTORCH_MODEL,
        onnx_output_path=ONNX_OUTPUT,
        model_name=MODEL_NAME,
        input_size=(1, 3, 112, 112),
        opset_version=11
    )

    if not success:
        print("\n❌ ONNX 변환 실패. 종료합니다.")
        return

    # Step 2: Calibration Dataset 준비
    if not os.path.exists(CALIBRATION_DATASET):
        print(f"\nCalibration dataset가 없습니다. 생성합니다: {CALIBRATION_DATASET}")
        success = prepare_calibration_dataset(
            lfw_dir=LFW_DIR,
            output_dir=CALIBRATION_DATASET,
            num_samples=NUM_CALIBRATION_SAMPLES
        )

        if not success:
            print("\n❌ Calibration dataset 준비 실패. 종료합니다.")
            return
    else:
        print(f"\n✅ Calibration dataset 이미 존재: {CALIBRATION_DATASET}")

    # Step 3: Calibration Config 생성
    success = generate_calibration_config(
        dataset_path=CALIBRATION_DATASET,
        config_output_path=CALIBRATION_CONFIG,
        num_samples=NUM_CALIBRATION_SAMPLES
    )

    if not success:
        print("\n❌ Calibration config 생성 실패. 종료합니다.")
        return

    # Step 4: DXNN 변환 안내
    print_dxnn_conversion_instructions(
        onnx_path=ONNX_OUTPUT,
        config_path=CALIBRATION_CONFIG,
        dxnn_output=DXNN_OUTPUT
    )

    print("\n" + "=" * 80)
    print("변환 프로세스 완료!")
    print("=" * 80)

    print("\n요약:")
    print(f"  ✅ ONNX 모델: {ONNX_OUTPUT}")
    print(f"  ✅ Calibration dataset: {CALIBRATION_DATASET}")
    print(f"  ✅ Calibration config: {CALIBRATION_CONFIG}")
    print(f"  ⏳ DXNN 변환: DeepX 컴파일러로 수동 실행 필요")

    print("\n다음 단계:")
    print("  1. DeepX 컴파일러로 ONNX -> DXNN 변환")
    print(f"  2. 변환된 DXNN 모델 테스트")
    print(f"  3. EdgeFaceNPURecognizer에서 사용")


if __name__ == "__main__":
    main()
