"""
tf_App_openvino.py  ─  OpenVINO IR(.xml/.bin) 모델 사용
변경 사항 (vs tf_App_keras.py):
  - TensorFlow / Keras 완전 제거
  - openvino.runtime Core로 모델 로드 및 추론
  - CAM 히트맵: IR에 포함된 block5_conv3 출력 노드를 이름으로 직접 추출
  - VGG16 전처리(BGR 변환 + ImageNet mean 차감)를 NumPy로 직접 구현
  - 실행 디바이스: CPU (필요 시 "GPU" 로 변경)
"""

import streamlit as st
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from PIL import Image

# OpenVINO 런타임
from openvino.runtime import Core

# ── 한글 폰트 설정 ──
font_path_win   = "C:/Windows/Fonts/malgun.ttf"
font_path_linux = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(font_path_win):
    font_manager.fontManager.addfont(font_path_win)
    matplotlib.rc('font', family='Malgun Gothic')
elif os.path.exists(font_path_linux):
    font_manager.fontManager.addfont(font_path_linux)
    matplotlib.rc('font', family='NanumGothic')
else:
    matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 상수 ──
INPUT_IMG_SIZE = (224, 224)
NEG_CLASS      = 1
CLASSES        = ["정상", "불량"]
MODEL_XML      = "./weights/leather_model.xml"
MODEL_BIN      = "./weights/leather_model.bin"
HEATMAP_THRES  = 0.5
DEVICE         = "CPU"   # Intel GPU 사용 시 "GPU" 로 변경

# ImageNet BGR mean (VGG16 keras preprocess_input 동일 값)
_IMAGENET_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)  # B, G, R

# block5_conv3 출력 노드 이름
# openvino.Model.outputs 에서 확인 후 필요 시 수정
FEAT_OUTPUT_NAME = "block5_conv3"   # 부분 문자열 매칭으로 탐색


# ─────────────────────────────────────────────
# 1. 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:", layout="wide")
st.title("InspectorsAlly")
st.caption("AI 기반 자동 검사로 품질 관리를 한 단계 높이세요")
st.write("제품 이미지를 업로드하면 AI 모델이 **정상 / 불량** 여부를 자동으로 판별합니다.")

with st.sidebar:
    if os.path.exists("./docs/overview_dataset.jpg"):
        st.image(Image.open("./docs/overview_dataset.jpg"))
    st.subheader("InspectorsAlly 소개")
    st.write(
        "InspectorsAlly는 기업의 품질 관리 검사를 효율화하기 위해 설계된 "
        "AI 기반 검사 애플리케이션입니다. VGG16 전이학습 기반으로 "
        "가죽 제품의 스크래치, 찍힘, 변색 등의 결함을 감지합니다."
    )
    st.divider()
    st.write("**모델 정보**")
    st.write(f"- 프레임워크: OpenVINO Runtime")
    st.write(f"- 백본: VGG16 (ImageNet 사전학습, 전체 동결)")
    st.write(f"- 출력: sigmoid 단일값 (0=정상, 1=불량)")
    st.write(f"- 입력 크기: {INPUT_IMG_SIZE[0]}×{INPUT_IMG_SIZE[1]}")
    st.write(f"- 실행 디바이스: {DEVICE}")


# ─────────────────────────────────────────────
# 2. 모델 로드
#    Core().compile_model() 로 xml+bin 동시 로드
#    CAM용 중간 출력(block5_conv3)을 추가 출력으로 지정
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_XML) or not os.path.exists(MODEL_BIN):
        return None, None, None

    ie    = Core()
    model = ie.read_model(model=MODEL_XML, weights=MODEL_BIN)

    # block5_conv3 출력 노드 탐색 (이름에 FEAT_OUTPUT_NAME 포함 여부로 탐색)
    feat_tensor = None
    for op in model.get_ordered_ops():
        if FEAT_OUTPUT_NAME in op.get_friendly_name():
            feat_tensor = op.output(0)
            break

    if feat_tensor is None:
        # 탐색 실패 시 CAM 비활성화 (예측만 수행)
        compiled = ie.compile_model(model=model, device_name=DEVICE)
        return compiled, None, None

    # 기존 출력(predictions) + block5_conv3 출력을 동시에 내보내도록 설정
    from openvino.runtime import Output
    original_outputs = model.outputs          # [predictions 출력]
    model.add_outputs([feat_tensor])          # CAM용 출력 추가
    compiled = ie.compile_model(model=model, device_name=DEVICE)

    # 출력 인덱스 확인
    #   compiled.outputs[0] → predictions (sigmoid)
    #   compiled.outputs[1] → block5_conv3 feature map
    # (모델에 따라 순서가 다를 수 있으므로 이름으로 재확인)
    pred_idx, feat_idx = 0, 1
    for i, out in enumerate(compiled.outputs):
        name = out.get_any_name()
        if FEAT_OUTPUT_NAME in name:
            feat_idx = i
        else:
            pred_idx = i

    return compiled, pred_idx, feat_idx


# ─────────────────────────────────────────────
# 3. 이미지 전처리 (TF 없이 NumPy로 직접 구현)
#    keras.applications.vgg16.preprocess_input 동일 동작:
#      RGB → BGR 변환 후 ImageNet mean 차감
# ─────────────────────────────────────────────
def preprocess_image(pil_img):
    img       = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)          # (224, 224, 3)  RGB

    # RGB → BGR
    img_bgr = img_array[..., ::-1].copy()

    # ImageNet mean 차감
    img_bgr -= _IMAGENET_MEAN

    # (H, W, C) → (1, H, W, C)  ← NHWC (TF/Keras 기본 레이아웃)
    return np.expand_dims(img_bgr, axis=0)


# ─────────────────────────────────────────────
# 4. CAM 히트맵 생성
# ─────────────────────────────────────────────
def generate_heatmap(compiled, pred_idx, feat_idx, img_array):
    """
    compiled  : compile_model 결과
    pred_idx  : predictions 출력 인덱스
    feat_idx  : block5_conv3 출력 인덱스 (None 이면 CAM 불가)
    """
    infer_req = compiled.create_infer_request()
    infer_req.infer({0: img_array})

    prob = float(infer_req.get_output_tensor(pred_idx).data.flatten()[0])
    class_idx = 1 if prob > 0.5 else 0

    if feat_idx is None:
        # CAM 불가 → 빈 히트맵 반환
        heatmap = np.zeros(INPUT_IMG_SIZE, dtype=np.float32)
        return heatmap, prob, class_idx

    feature_maps = infer_req.get_output_tensor(feat_idx).data  # (1, H, W, 512) or (1, 512, H, W)

    # 레이아웃 확인 후 (H, W, C) 형태로 정규화
    fm = feature_maps[0]
    if fm.shape[0] == 512:           # NCHW 레이아웃
        fm = fm.transpose(1, 2, 0)  # → (H, W, 512)

    # CAM 가중치: dense + predictions 레이어 가중치를 직접 IR에서 가져올 수 없으므로
    # 채널 축 평균으로 근사 (Grad-CAM 없이 사용 가능한 최선의 근사)
    # ※ 정확한 CAM이 필요하면 학습 후 가중치를 별도 npy로 저장해 두는 것을 권장
    cam = fm.mean(axis=-1)           # (H, W)

    cam_min, cam_max = cam.min(), cam.max()
    norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    heatmap_pil     = Image.fromarray((norm_cam * 255).astype(np.uint8))
    heatmap_resized = np.array(heatmap_pil.resize(INPUT_IMG_SIZE)) / 255.0
    return heatmap_resized, prob, class_idx


def get_bbox_from_heatmap(heatmap, thres=0.5):
    binary_map = heatmap > thres
    if not binary_map.any():
        return None
    x_dim  = np.max(binary_map, axis=0) * np.arange(binary_map.shape[1])
    y_dim  = np.max(binary_map, axis=1) * np.arange(binary_map.shape[0])
    x_vals = x_dim[x_dim > 0]
    y_vals = y_dim[y_dim > 0]
    if len(x_vals) == 0 or len(y_vals) == 0:
        return None
    return int(x_vals.min()), int(y_vals.min()), int(x_dim.max()), int(y_dim.max())


# ─────────────────────────────────────────────
# 5. 결과 시각화
# ─────────────────────────────────────────────
def visualize_result(pil_img, heatmap, class_idx, prob, thres=HEATMAP_THRES):
    img_np = np.array(pil_img.resize(INPUT_IMG_SIZE).convert("RGB"))

    if class_idx == NEG_CLASS:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        axes[0].imshow(img_np)
        axes[0].set_title("원본 이미지", fontsize=11)
        axes[0].axis("off")
        axes[1].imshow(img_np)
        axes[1].imshow(heatmap, cmap="Reds", alpha=0.45)
        axes[1].set_title(f"불량 감지 히트맵 (불량 확률: {prob:.3f})", fontsize=11)
        axes[1].axis("off")
        bbox = get_bbox_from_heatmap(heatmap, thres)
        if bbox:
            x0, y0, x1, y1 = bbox
            rect = mpatches.Rectangle(
                (x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor="red", facecolor="none"
            )
            axes[1].add_patch(rect)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(img_np)
        ax.set_title(f"정상 (불량 확률: {prob:.3f})", fontsize=11)
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


# ─────────────────────────────────────────────
# 6. 메인 UI
# ─────────────────────────────────────────────
compiled, pred_idx, feat_idx = load_model()

if compiled is None:
    st.error(
        f"모델 파일을 찾을 수 없습니다.\n\n"
        f"필요한 파일:\n"
        f"- `{MODEL_XML}`\n"
        f"- `{MODEL_BIN}`\n\n"
        "Keras 모델을 변환하려면 아래 명령을 실행하세요:\n\n"
        "```bash\n"
        "mo --input_model weights/leather_model.keras \\\n"
        "   --output_dir weights/ \\\n"
        "   --model_name leather_model\n"
        "```"
    )
    st.stop()

st.subheader("이미지 입력 방법 선택")
input_method = st.radio("options", ["파일 업로드", "카메라 촬영"],
                        label_visibility="collapsed")
pil_image = None

if input_method == "파일 업로드":
    uploaded_file = st.file_uploader("이미지 파일을 선택하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.image(pil_image, caption="업로드된 이미지", width=300)
        st.success("이미지가 성공적으로 업로드되었습니다!")
    else:
        st.warning("검사할 이미지 파일을 업로드해주세요.")

elif input_method == "카메라 촬영":
    st.warning("카메라 접근 권한을 허용해주세요.")
    camera_file = st.camera_input("카메라로 이미지 촬영")
    if camera_file:
        pil_image = Image.open(camera_file).convert("RGB")
        st.image(pil_image, caption="촬영된 이미지", width=300)
        st.success("이미지가 성공적으로 촬영되었습니다!")
    else:
        st.warning("카메라로 이미지를 촬영해주세요.")

submit = st.button(label="가죽 제품 이미지 검사 시작", type="primary")

if submit:
    if pil_image is None:
        st.error("이미지를 먼저 업로드하거나 카메라로 촬영해주세요.")
    else:
        st.subheader("검사 결과")
        with st.spinner("이미지를 분석 중입니다..."):
            img_array                = preprocess_image(pil_image)
            heatmap, prob, class_idx = generate_heatmap(compiled, pred_idx, feat_idx, img_array)
        label = CLASSES[class_idx]
        if label == "정상":
            st.success(f"✅ **정상** (불량 확률: {prob:.1%})\n\n제품 검사 결과 이상이 감지되지 않았습니다.")
        else:
            st.error(
                f"⚠️ **불량 감지** (불량 확률: {prob:.1%})\n\n"
                "아래 히트맵에서 결함이 의심되는 영역(빨간 박스)을 확인하세요."
            )
        st.write("**검사 결과 시각화**")
        visualize_result(pil_image, heatmap, class_idx, prob)
        st.write("**클래스별 예측 확률**")
        col1, col2 = st.columns(2)
        col1.metric("정상", f"{(1 - prob):.1%}")
        col2.metric("불량", f"{prob:.1%}")
        st.progress(float(prob), text=f"불량 확률: {prob:.1%}")