from rag_system import create_rag_system
from data import get_vectorstore

def medical_info():
    # 입력 가이드
    def get_input(prompt, allow_empty=False):
        while True:
            user_input = input(prompt).strip()
            if user_input.lower() == 'q':
                return None
            if user_input or allow_empty:
                return user_input
            print("값을 입력해주세요.")
            
    while True:
        sex = get_input("성별을 입력해주세요. (중단은 'q' 입력, 모르면 enter) :").strip()
        if sex in None:
            return None
        if sex in ["남성", "여성", " "]:
            sex = sex if sex else "미확인"
            break
        print("남성, 여성으로 입력해주세요.")
        
    while True:
        symptoms = get_input("증상을 입력해주세요. (중단은 'q' 입력, 모르면 enter) :").strip()
        if symptoms.lower() == 'q':
            return None
        if symptoms:
            break
        print("증상을 입력해주세요.")
        
    while True:
        vital_signs = get_input("활력징후를 입력해주세요.(혈압-맥박-spo2-체온-혈당 순) (중단은 'q' 입력, 모르면 enter) :").strip()
        if vital_signs.lower() == 'q':
            return None
        if vital_signs:
            break
        print("활력징후를 입력하세요.")
        
    while True:
        consciousness = get_input("의식상태를 입력해주세요. (중단은 'q' 입력, 모르면 enter) :").strip()
        if consciousness.lower() == 'q':
            return None
        if consciousness:
            break
        print("의식상태를 입력하세요.")
            
    age = get_input("나이를 입력해주세요. (생략가능)", allow_empty=True) or "미확인"
    diseases = get_input("기저질환을 입력해주세요. (생략가능)", allow_empty=True) or "미확인"
    medications = get_input("복용약물을 입력해주세요. (생략가능)", allow_empty=True) or "미확인"
    
    # 수집된 정보 딕셔너리 변환
    return {
        "sex" : sex,
        "age" : age,
        "symptoms": symptoms,
        "vital_signs": vital_signs,
        "consciousness": consciousness,
        "diseases": diseases,
        "medications": medications
    }
    
    
def main():
    vectorstore = get_vectorstore()
    
    # rag 시스템 생성
    ktas_chain = create_rag_system(vectorstore)
    
    # 대화형 인터페이스
    while True:
        # 정보 수집
        user_info = medical_info()
        
        if user_info is None:
            print("종료")
            return None
        
        response = ktas_chain.invoke({
            "input" : user_info["symptoms"],
            "sex" : user_info["sex"],
            "age": user_info["age"],
            "vital_signs": user_info["vital_signs"],
            "consciousness": user_info["consciousness"],
            "underlying_diseases": user_info["underlying_diseases"],
            "medications": user_info["medications"]
        })
        
        print("====답변====")
        print(response["answer"])
        
        
if __name__ == "__main__":
    main()
    
