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
    for i in range(1):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_container_width=1000)
    # 2nd Row - YouTube Videos
    for i in range(1):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(2):
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
            "https://i.ibb.co/B2HjP1W/image.webp",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=x-PyGv9_CJ4",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 영국(스코틀랜드)으로 성격은 온순하며 대형견이다. 수컷은 61cm에 34kg까지 자라며 암컷은 56cm에 30kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/GT8zJRN/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=NBT2QrulQp8",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 독일로 성격은 사람을 잘 따른며 소형견이다. 수컷은 27cm 암컷은 24cm까지 자라며 암수 둘다 12kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[2]: {
        'images': [
            "https://i.ibb.co/Xbqv83w/image.jpg",
            "https://via.placeholder.com/300?text=Label3_Image2",
            "https://via.placeholder.com/300?text=Label3_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=eJvcoohXDuY",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "원산지는 독일로 성격은 온순하고 호기심이 왕성하며 대형견이다. 수컷은 72cm에 45kg까지 자라며 암컷은 68cm에 35kg까지 자란다.",
            "Label 3 관련 두 번째 텍스트 내용입니다.",
            "Label 3 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[3]: {
        'images': [
            "https://i.ibb.co/rw3CkYc/image.jpg",
            "https://via.placeholder.com/300?text=Label3_Image2",
            "https://via.placeholder.com/300?text=Label3_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=FABOw10lRbg",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "원산지는 영국으로 성격은 온순하고 매우 영리하고 사람을 좋아하며 대현견이다. 수컷은 57cm에 34kg까지 자라며 암컷은 56cm에 32kg까지 자란다.",
            "Label 4 관련 두 번째 텍스트 내용입니다.",
            "Label 4 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[4]: {
        'images': [
            "https://i.ibb.co/KL5Xq2M/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=9xFqDi0HOq8",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 유럽(몰타)으로 성격은 온화하고 지능이 높으며 소형견이다. 25cm에 3.3kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[5]: {
        'images': [
            "https://i.ibb.co/R7KyyzQ/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=ghqEL4cP5yc",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 독일로 성격은 천진난만하고 호기심이 가득하며 소형견이다. 35cm에 7kg까지 자란다",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[6]: {
        'images': [
            "https://i.ibb.co/6X2qrH4/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=UKapx99VQcw",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 독일로 성격은 쾌활하고 활발하지만 다소 신경질적이며 소형견이다. 30cm에 6kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[7]: {
        'images': [
            "https://i.ibb.co/Gn405Wx/qpemffldxjs-xpfldj.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=_5OEV4P1_HU",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 유럽(영국)으로 성격은 호기심이 왕성하고 주인에게는 애정이 깊지만, 섬세하고 신경질적이며 중형견이다. 수컷은 41cm 암컷은 41cm이하 정도로 자라며 암수모두 10.4kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[8]: {
        'images': [
            "https://i.ibb.co/HnjvrbN/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=Wf_A_cnwwvk",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 벨기에와 프랑스로 성격은 명량하고 다정하고 감수성이 풍부하며 소형견이다. 30cm에 6kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[9]: {
        'images': [
            "https://i.ibb.co/7yKPt4G/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=E8FPnTQxFMo",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 러시아 북부 및 시베리아로 성격은 사람을 잘 따르고 외로움을 많이타며 중형견이다. 수컷은 60cm 암컷은 56cm까지 자라며 암수 둘다 30kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[10]: {
        'images': [
            "https://i.ibb.co/vmDzH8x/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=2tNFHa3LWT4",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 대한민국으로 성격은 인내심이 강하고 보호자에게 충직하며 중형견이다. 60cm에 32kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[11]: {
        'images': [
            "https://i.ibb.co/YfLT8mk/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=6fSX35yz8_M",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 독일로 성격은 높은 지능을 가지고 있고 주인에게 충성하며 대형견이다. 수컷은 65cm에 40kg까지 자라며 암컷은 60cm에 32kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[12]: {
        'images': [
            "https://i.ibb.co/23RhrF3/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=9xmqNrpO8bE",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 미국이며 성격을 쾌활하고 낙천적이고 주인에게 충성스러우며 대형견이다. 수컷은 60cm에 28kg까지 자라며 암컷은 56cm에 23kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[13]: {
        'images': [
            "https://i.ibb.co/0rfH8FP/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=xZ0fO7PLY9M",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 중국으로 성격은 나이브하면서도 감정이나 표정은 오버하는 경향이 있으며 소형견이다. 26.7cm이하 8.1kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[14]: {
        'images': [
            "https://i.ibb.co/g4zX3TN/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3dkQJpO8eYI",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 미국(알래스카 지방)으로 성격은 온순하고 조용하며 대형견이다. 수컷은 63.5cm에 38kg까지 자라며 암컷은 58.5cm에 34kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[15]: {
        'images': [
            "https://i.ibb.co/58zC94Z/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=IL7AlqgxQng",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 대한민국으로 현재는 그 실체가 불분명한 개로 복원 사업 중이나 여럿 논란이 있는 상태이다. 다만 2024년 8월 30일 전라북도 임실군에 따르면 유엔 식량농업기구(FAO) 가축다양성정보시스템에 오수개가 정식 품종으로 등재되었다고 밝혔다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[16]: {
        'images': [
            "https://i.ibb.co/hWdq9pn/dyzmtu-xpfldj.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=mgRcG6Z5kBY",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 영국으로 성격은 주인과 있으면 드세고 쾌활해지며 소형견이다. 23cm에 3.1kg이내까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[17]: {
        'images': [
            "https://i.ibb.co/XVt6bdj/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=mXO3DfUgj0I",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 유럽(영국)으로 성격은 자신감있고 친절하며 중형견이다. 30cm에 17kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[18]: {
        'images': [
            "https://i.ibb.co/mzVNQkD/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=Iw0ElufMEsU&t=17s",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 이탈리아로 성격은 다정하고 애정이 깊지만 겁이 많으며 소형견이다. 38cm에 5kg이하까지 자란다",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[19]: {
        'images': [
            "https://i.ibb.co/pbRGGy2/wovoslwm-tmvlcm.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=6vvszdv5pDY",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 일본으로 성격은 쾌활하고 놀기 좋아하지만 낯가림이 심하며 소형견이다. 수컷은 38cm자라고 암컷은 수컷보다 약간 작으며 암수 모두 6kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[20]: {
        'images': [
            "https://i.ibb.co/p3CGHn4/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=NHGkLb5ifu4",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 영국으로 성격은 매우 활발하고 겁이 없으며 장난을 좋아하며 소형견이다. 30cm에 6kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[21]: {
        'images': [
            "https://i.ibb.co/DLZ6wdK/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=vW6Ry2kdsBc&t=336s",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 대한민국으로 성격은 영리하고 활동적이며 중형견이다. 53cm에 20.8kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[22]: {
        'images': [
            "https://i.ibb.co/XbyLSTF/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=02rMn4giIUs",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 맥시코로 성격은 제멋대로이며 소형견이다. 23cm에 2.47kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[23]: {
        'images': [
            "https://i.ibb.co/tQnschQ/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3AY3xj_mpu4",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 프랑스, 벨기에로 성격은 영리하고 상황판단을 잘하지만 나이브하며 소형견이다. 28cm이하에 5kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[24]: {
        'images': [
            "https://i.ibb.co/NLzsv2c/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=j220tzwpeZo",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 중국으로 성격은 애교가 많고 우호적이며 소형견이다. 28cm에 8.1kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[25]: {
        'images': [
            "https://i.ibb.co/5xBZyWn/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=MItHENgKUpQ",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 독일로 성격은 쾌활하고 호기심이 왕성하지만 신결질적인 면도 있으며 소형견이다. 22cm에 2.3kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[26]: {
        'images': [
            "https://i.ibb.co/X8L09mj/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=xIE-G1TWEHk",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 프랑스로 성격은 온순하고 쾌활하며 중형견이다. 60cm에 수컷은 25kg 전후 암컷은 23kg 전후까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[27]: {
        'images': [
            "https://i.ibb.co/pxVrTKD/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=LUhQZxZPQBo",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 대한민국으로 성격은 주인을 잘 따르고 용맹하고 영리하며 대견이다. 60cm에 30kg까지 자란다.",
            "Label 2 관련 두 번째 텍스트 내용입니다.",
            "Label 2 관련 세 번째 텍스트 내용입니다."
        ]
    },
    labels[28]: {
        'images': [
            "https://i.ibb.co/tb81PVs/image.jpg",
            "https://via.placeholder.com/300?text=Label2_Image2",
            "https://via.placeholder.com/300?text=Label2_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=--RH6FC_uXs",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g"
        ],
        'texts': [
            "원산지는 프랑스이며 성격은 호기심이 왕성하고 놀기 좋아하며 소형견이다. 30cm에 14kg까지 자란다.",
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
