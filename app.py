# (venv)환경에서 streamlit run app.py 명령어 실행
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_upstage import ChatUpstage
from src.data import get_vectorstore
from src.rag_system import create_rag_system, get_ktas_prompt
import os
from dotenv import load_dotenv

load_dotenv()

# 페이지 제목
st.set_page_config(page_title="KTAS 중증도 분류 시스템", layout="wide")

# 사이드바에서 앱 정보 표시
with st.sidebar:
    st.title("KTAS 중증도 분류")
    st.markdown("""
    이 프로그램은 KTAS(Korean Triage and Acuity Scale)를 사용하여 
    환자의 중증도를 평가합니다.
    
    **참고**: 이 평가는 참고용이며, 정확한 의료인의 판단이 필요합니다.
    """)

    # KTAS 정보 
    st.markdown("### KTAS 단계")
    st.markdown("""
    - **KTAS 1**: 즉각적인 소생술 필요 
    - **KTAS 2**: 고위험 상황
    - **KTAS 3**: 급성 질환
    - **KTAS 4**: 아급성/만성 상태 
    - **KTAS 5**: 비응급 상태
    """)

# 메인 페이지 헤더
st.title("환자 정보")

# 데이터 초기화
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {
         "sex": "",
        "age": "",
        "diseases": "",
        "medications": "",
        "vital_signs": "",
        "consciousness": "",
        "symptoms": ""
    }

if 'assessment_result' not in st.session_state:
    st.session_state.assessment_result = None

# 폼 생성
with st.form("patient_info_form"):
    # 2열 레이아웃
    col1, col2 = st.columns(2)

    with col1:
        # 환자 기본 정보
        st.subheader("기본 정보")
        sex = st.radio("성별", ["남성", "여성"], index=0)
        
        # 나이와 의식상태를 가로로 나란히 배치
        age_consciousness_cols = st.columns(2)
        with age_consciousness_cols[0]:
            age = st.text_input("나이", placeholder="예: 30")
        with age_consciousness_cols[1]:
            consciousness = st.selectbox("의식상태", 
                ["명료(Alert)", "언어자극에 반응(Verbal)", "통증자극에 반응(Pain)", "무반응(Unresponsive)"])

        vital_signs = st.text_input("활력징후", placeholder="혈압-맥박-산소포화도-체온-혈당 \n예: 120/80-75-100-36.5-80")
        st.caption("혈압-맥박-산소포화도-체온-혈당 순서로 입력해주세요.")

    with col2:
        # 의학적 배경
        st.subheader("현병력")
        diseases = st.text_area("기저질환", 
            placeholder="고혈압, 당뇨, 심장질환 등이 있으면 입력하세요.", 
            height=90)
        
        medications = st.text_area("복용약물", 
            placeholder="현재 복용 중인 약물을 입력하세요.", 
            height=90)
        
    # 증상
    st.subheader("현재 증상")
    symptoms = st.text_area("증상", height=100)
    
    # 제출 버튼
    submit_button = st.form_submit_button("중증도 평가")


# 폼제출
if submit_button:
    # 입력 데이터 저장
    st.session_state.patient_data = {
        "sex": sex,
        "age": age if age else "미확인",
        "diseases": diseases if diseases else "미확인",
        "medications": medications if medications else "미확인",
        "vital_signs": vital_signs,
        "consciousness": consciousness,
        "symptoms": symptoms
    }

    # 필수 입력 확인
    if not vital_signs or not symptoms:
        st.error("활력징후와 증상은 필수 입력 사항입니다.")
    else:
        with st.spinner("환자 정보 분석 중..."):
            # 1. 벡터스토어 연결
            vectorstore = get_vectorstore()
            
            # 2. RAG 시스템 생성
            ktas_chain = create_rag_system(vectorstore)
            
            # 3. 체인에 입력 데이터 전달
            response = ktas_chain.invoke({
                "input": st.session_state.patient_data["symptoms"],  # retriever 쿼리용
                "sex": st.session_state.patient_data["sex"],
                "age": st.session_state.patient_data["age"],
                "diseases": st.session_state.patient_data["diseases"],
                "medications": st.session_state.patient_data["medications"],
                "vital_signs": st.session_state.patient_data["vital_signs"],
                "consciousness": st.session_state.patient_data["consciousness"],
                "symptoms": st.session_state.patient_data["symptoms"]
            })
            
            # 4. 응답 처리 및 표시
            st.session_state.assessment_result = response["answer"]

# 결과 표시
if st.session_state.assessment_result:
    st.success("평가가 완료되었습니다!")
    
    # 결과 카드
    st.subheader("KTAS 평가 결과")
        
    # 실제 구현에서는 아래 하드코딩된 결과를 RAG 시스템의 출력으로 대체
    with st.container():
        st.markdown(st.session_state.assessment_result)

        # KTAS 점수 추출 및 적절한 경고 표시 (실제 응답에서 KTAS 점수 추출 로직 필요)
        if "KTAS 1" in st.session_state.assessment_result:
            st.error("⚠️ 이 환자는 즉시 의료진의 처치가 필요합니다!")
        elif "KTAS 2" in st.session_state.assessment_result:
            st.warning("⚠️ 이 환자는 15분 이내 의료진의 진찰이 필요합니다.")
        elif "KTAS 3" in st.session_state.assessment_result:
            st.info("이 환자는 30분 이내 의료진의 진찰이 필요합니다.")


# RAG 시스템과 연결하는 함수
def create_rag_system(vectorstore):
    # LLM 초기화
    llm = ChatUpstage(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="solar-pro"
    )
    
    # 프롬프트 정의
    prompt = get_ktas_prompt()
    
    # 문서 체인 생성
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # retriever 생성
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={"k":3}
    )
    
    # 최종 검색-생성 체인 생성
    retriever_chain = create_retrieval_chain(
        retriever, 
        document_chain,
        input_key="input"  # 증상을 검색 쿼리로 사용
    )

    return retriever_chain
