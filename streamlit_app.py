import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1I-bafgaRgOnsMVcHCOVASjo1Ng3XbnrL'

# Google Drive에서 파일 다운로드 함수
@st.cache_resource
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 종 분류 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_container_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 관련 자료 및 설명")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_container_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://ibb.co/WpmyQyW"
            "https://via.placeholder.com/300?text=Label1_Image2",
            "https://via.placeholder.com/300?text=Label1_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "골든 리트리버",
            "Label 1 관련 두 번째 텍스트 내용입니다.",
            "Label 1 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[1]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "닥스훈트",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[2]: {
        'images': [
            "https://via.placeholder.com/300?text=Label3_Image1",
            "https://via.placeholder.com/300?text=Label3_Image2",
            "https://via.placeholder.com/300?text=Label3_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "도베르만 핀셔",
            "Label 3 관련 두 번째 텍스트 내용입니다.",
            "Label 3 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[3]: {
        'images': [
            "https://via.placeholder.com/300?text=Label3_Image1",
            "https://via.placeholder.com/300?text=Label3_Image2",
            "https://via.placeholder.com/300?text=Label3_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "래브라도 리트리버",
            "Label 4 관련 두 번째 텍스트 내용입니다.",
            "Label 4 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[4]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "말티즈",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[5]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "미니어쳐 슈나우저",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[6]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "미니어쳐 핀셔",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[7]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "베들링턴 테리어",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[8]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "비숑프리제",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[9]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "사모예드",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[10]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "삽살개개",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[11]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "셰퍼드",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[12]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "시베리안 허스키",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[13]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "시추",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[14]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "알래스탄 맬러뮤트",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[15]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "오수개",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[16]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "요크셔 테리어",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[17]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "웰시코기",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[18]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "이탈리안 그레이 하운드",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[19]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "재페니스 스피츠",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[20]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "잭 러셀 테리어",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[21]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "진도개",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[22]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "치와와",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[23]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "파피용",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[24]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "퍼그",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[25]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "포메라니안",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[26]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "푸들",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[27]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "풍산개",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[28]: {
        'images': [
            "https://via.placeholder.com/300?text=Label2_Image1",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "프렌치 불독",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)
    probs = probs.tolist()  # 텐서를 리스트로 변환

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)
