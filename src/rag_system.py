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
    return ChatPromptTemplate.from_message([
        # system prompt
        ("system")
    ])