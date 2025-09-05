# 김광무 개발 소스 폴더

# VLM 사용 OCR 처리 : https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
* 20250903_SEMI_Qwen2.5VLM_testing.ipynb : VLM을 이용하여 1개 이미지 OCR 추론 샘플 테스트 코드
* 20250904_SEMI_Qwen2.5VLM_OCR처리.ipynb : testing 코드를 기반하여 수집된 62개 이미지를 한꺼번에 추론하여 결과를 [원본파일명]_ocr.txt 로 저장하는 코드
* 20250904_SEMI_Qwen2.5VLM_OCR처리_2.5재시도.ipynb : 알고보니 위 파일은 2.5가 아닌 2로 돌린 결과라서 2.5로 수정후 다시돌림 ocr25.txt 생성
* diff_sh.sh : Qwen2와 Qwen2.5의 생성 내용 비교용 shell script
* diff_results.log : diff 결과 2.5가 인식 결과가 좋음
* 20250905_SEMI_Qwen2.5_VLM_OCR처리_개선.ipynb : OCR 처리 결과를 바로 19개 분류하려고 했으나 실패한 버전
* 20250905_SEMI_Qwen2.5_VLM_OCR처리_개선2.ipynb : OCR 처리 결과를 원재료명과 혼입,같은제조공정 으로 매끄럽게 다듦은 버전
* 20250905_SEMI_Qwen2.5_VLM_식품성분분류.ipynb : OCR추출된 new.txt 파일을 기반으로 알러지유발 원재료명을 19가지로 분류하는 프롬프트 적용 코드 파일은 _new_result.txt로 생성
