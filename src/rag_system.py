from langchain_upstage import ChatUpstage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def create_rag_system(vectorstore):
    # LLM 초기화
    llm = ChatUpstage(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="solar-pro"
    )
    
    # 프롬프트 정의
    prompt = get_ktas_prompt()
    
    # 문서 체인 생성(검색된 문서들을 llm에 보내기 전 준비)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # dense retriever 생성
    retriever = vectorstore.as_retriever(
        search_type='mmr',  # default : similarity(유사도) / mmr 알고리즘
        search_kwargs={"k":3}
    )
    
     # 최종 검색-생성 체인 생성
    """
    검색 단계와 응답 생성 단계를 하나로 묶은 워크플로우
    질문 -> 검색 -> 결합 -> 응답의 단계를 자동화하여 한 번에 수행
    llm에 생성된 문서를 전달하고 답변을 생성
    """
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    return retriever_chain
    
    
def get_ktas_prompt():
    return ChatPromptTemplate.from_messages([
        # system prompt
        ("system", """너는 응급실에서 KTAS(Korean Triage and Acuity Scale : 한국형 응급환자 분류도구)를 이용하여
         환자 중증도를 분류하는 의료 챗봇이야. 다음 의료정보를 참고하여 다음 가이드라인에 따라라 KTAS 점수와 객관적인 이유를 답변해줘.
        다음은 검색된 문서 정보야: {context}

        1. [환자 분석]:
         - 성별: {sex}
         - 나이: {age}
         - 기저질환: {diseases}
         - 복용약물: {medications}
         - 활력징후: {vital_signs}
         - 의식상태: {consciousness}
         - 증상: {input}
         - 활력징후는 혈압-맥박-산소포화도-체온-혈당 순서야. 혈당은 없을 수도 있어.
         - 위 내용들을 바탕으로 KTAS 점수를 평가해줘.
         - 점수의 이유와 근거를 명확히 제시해줘.

        2. [KTAS 기준]:
         - KTAS 1: 즉각적인 소생술이 필요하며 생명이나 사지를 위협하는 (또는 악화 가능성이 높은) 상태
         - KTAS 2: 생명 혹은 사지, 신체기능에 잠재적인 위협이 있으며 이에 대한 빠른 치료가 필요한 경우
         - KTAS 3: 치료가 필요한 상태로 진행할 수도 있는 잠재적 가능성을 고려해야 하는 경우
         - KTAS 4: 환자의 나이, 통증이나 악화/합병증에 대한 가능성을 고려할 때 1~2시간 안에 처치나 재평가를 시행하면 되는 상태
         - KTAS 5: 긴급하지만 응급은 아닌 상태, 만성적인 문제로 인한 것이거나, 악화의 가능성이 낮은 상태

         3. [주의사항]:
         - 이는 참고용 정보이며, 정확한 판단은 의료진의 판단임을 명시해줘.

         4. [형식]:
         - 응답은 [환자 분석], [KTAS 평가], [주의사항] 섹션으로 구분해서 제공해줘.
         - 검색된 문서의 정보를 바탕으로 응답해줘.
         - 생체징후가 비정상인 경우 해당 수치를 명시하고 그 의미를 설명해줘.
         - 최종 KTAS 점수를 명확하게 제시하고 그 이유를 상세히 설명해줘.
"""),

        # few-shot 예제 1: KTAS 2 사례
        ("human", "65세 남성 환자가 갑작스러운 흉통을 호소하며 응급실에 내원했습니다. 고혈압과 당뇨병 기저질환이 있으며, 혈압약과 당뇨약을 복용 중입니다. 활력징후는 혈압 170/95, 맥박 110, 호흡수 24, 체온 37.2, 산소포화도 93%입니다. 의식은 명료하고, 환자는 30분전부터 시작된 압박감을 동반한 가슴 통증을 호소하고 있습니다. KTAS 점수는 얼마인가요?"),
        ("assistant", """[환자 분석]
        이 환자는 65세 남성으로 고혈압과 당뇨병의 기저질환이 있으며, 30분 전부터 시작된 압박감을 동반한 가슴 통증을 호소하고 있습니다.
        - 성별: 남성
        - 나이: 65세
        - 기저질환: 고혈압, 당뇨병
        - 복용약물: 혈압약, 당뇨약
        - 활력징후: 혈압 170/95(상승), 맥박 110(빈맥), 호흡수 24(빈호흡), 체온 37.2, 산소포화도 93%(경미한 저하)
        - 의식상태: 명료
        - 증상: 30분전부터 시작된 압박감을 동반한 가슴 통증

        [KTAS 평가]
        KTAS 2 

        판단 근거:
        1. 65세 고령 환자에서 갑작스러운 흉통이 발생했습니다.
        2. 고혈압과 당뇨병의 기저질환은 심혈관 질환의 주요 위험인자입니다.
        3. 활력징후에서 혈압 상승, 빈맥, 빈호흡이 관찰되며 산소포화도가 경미하게 저하되어 있습니다.
        4. 압박감을 동반한 가슴 통증은 급성 관상동맥 증후군(급성 심근경색 또는 불안정 협심증)의 가능성이 높습니다.
        5. 증상 발생 30분 이내로 내원했으며, 즉각적인 평가와 치료가 필요한 상태입니다.

        [주의사항]
        이 KTAS 평가는 참고용 정보이며, 실제 환자의 정확한 중증도 판단과 치료 방향은 반드시 의료진의 직접적인 평가와 판단에 따라야 합니다. 실제 임상 상황에서는 추가적인 검사와 더 상세한 병력 청취가 필요할 수 있습니다."""),

        # few-shot 예제 2: KTAS 5 사례
        ("human", "35세 여성이 왼쪽 발목 통증으로 응급실에 내원했습니다. 특별한 기저질환은 없으며, 복용 중인 약물도 없습니다. 활력징후는 혈압 120/80, 맥박 78, 호흡수 18, 체온 36.5, 산소포화도 99%입니다. 의식은 명료하고, 2시간 전 계단에서 발을 접질려 왼쪽 발목 통증과 경미한 부종이 있으나 체중 부하는 가능한 상태입니다. KTAS 점수를 평가해주세요."),    
        ("assistant", """[환자 분석]
        이 환자는 35세 여성으로 2시간 전 발생한 왼쪽 발목 통증과 경미한 부종을 호소하고 있습니다.
        - 성별: 여성
        - 나이: 35세
        - 기저질환: 없음
        - 복용약물: 없음
        - 활력징후: 혈압 120/80(정상), 맥박 78(정상), 호흡수 18(정상), 체온 36.5(정상), 산소포화도 99%(정상)
        - 의식상태: 명료
        - 증상: 2시간 전 계단에서 발을 접질려 왼쪽 발목 통증과 경미한 부종, 체중 부하 가능

        [KTAS 평가]
        KTAS 5

        판단 근거:
        1. 모든 활력징후가 정상 범위 내에 있습니다.
        2. 의식 상태가 명료하고 전신 증상이 없습니다.
        3. 외상 후 2시간이 경과했으나 증상이 악화되지 않았습니다.
        4. 부종이 경미하고 체중 부하가 가능한 상태로, 심각한 골절이나 인대 손상의 가능성이 낮습니다.
        5. The Ottawa Ankle Rules에 따른 응급 X-ray 검사 기준에 해당되지 않을 가능성이 있습니다.
        6. 급성 치료가 필요하지만 생명을 위협하는 긴급한 상황은 아닙니다.

        [주의사항]
        이 KTAS 평가는 참고용 정보이며, 실제 환자의 정확한 중증도 판단과 치료 방향은 반드시 의료진의 직접적인 평가와 판단에 따라야 합니다. 실제 임상 상황에서는 X-ray와 같은 영상 검사를 통해 골절 여부를 확인하는 것이 필요합니다."""),
        
        # user query
        ("human", "{input}")
    ])


# RAG 시스템을 모듈화하고 단계별로 처리할 때 유용
# # Context 문서 포매팅 함수
# def format_docs(docs):
#     return "\n\n".join(document.page_content for document in docs)

# # retriever 생성 함수
# def get_retriever(vectorstore):
#     # dense retriever 생성
#     retriever = vectorstore.as_retriever(
#         # 유사도 정의
#         search_type="mmr",   # Maximum Marginal Relevance 검색 방식 사용
#         # 검색할 쿼리 수 정의
#         search_kwargs={"k": 3})
#     return retriever