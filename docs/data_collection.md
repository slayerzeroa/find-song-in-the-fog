# 음원 데이터 수집/적재 전략

이 문서는 `허밍 검색(QBH)` 관점에서 기존 음원 데이터를 현실적으로 수집하고, 검색 인덱스로 적재하는 방법을 정리합니다.

## 1) 원칙

- 원본 음원을 검색 시스템에 장기 저장하지 않는다.
- 인덱스에는 `검색용 표현`(멜로디 임베딩, fingerprint)만 저장한다.
- 권리/라이선스 범위가 명확한 소스만 수집한다.

## 2) 수집 소스 우선순위

1. 오픈 라이선스/연구용 데이터셋
2. 정식 계약된 B2B 카탈로그(유통사/라벨/플랫폼)
3. 사용자 업로드(명시적 동의 + 저작권 정책 적용)

초기 MVP에서는 1번으로 모델/인덱스 품질을 검증하고, 상용 단계에서 2번으로 확장하는 구성이 안전합니다.

## 3) 권장 데이터셋(개발/검증)

- Free Music Archive(FMA): 오픈 라이선스 기반
- Jamendo/Open catalogs: 라이선스 확인 가능한 트랙 확보 용이
- MIR 벤치마크 계열(학술용): 성능 회귀 테스트에 유용

주의: 각 데이터셋마다 `상업적 사용`, `재배포`, `파생 인덱스 보관` 허용 범위가 다릅니다. 수집 전에 라이선스를 반드시 기록하세요.

### 확인된 제약 (2026-02 기준)

- FMA: 메타데이터(CC BY 4.0)와 오디오 권리 주체가 다르며, 오디오는 아티스트가 선택한 라이선스를 트랙별로 확인해야 함
- MTG-Jamendo: 비상업 연구용 중심이며, 상업 사용은 Jamendo 별도 허가 필요
- AcoustID Web Service: 무료 공개 엔드포인트는 비상업/속도 제한(초당 3req) 가이드 존재
- Spotify 개발자 정책: 플랫폼/콘텐츠를 ML/AI 학습에 사용하는 행위 금지
- YouTube API 정책: YouTube 시청각 콘텐츠 다운로드/백업/저장 금지

즉, 스트리밍 플랫폼 API를 직접 크롤링해 검색 인덱스를 구축하는 방식은 정책 위반 가능성이 높아 권장하지 않습니다.

## 4) 오프라인 적재 파이프라인

1. 메타데이터 수집: track_id, title, artist, source, license, ingest_date
2. 오디오 접근: 원본/프리뷰 파일 로딩
3. 구간 샘플링: 곡당 N개(예: 4개) 멜로디 구간 추출
4. 임베딩 생성: `melody_embedding.py`
5. ANN 인덱싱: `main.py index-ann`으로 FAISS(IVFPQ/HNSW)
6. 검증/등록: 품질 지표(Top-1, Top-5, MRR) 통과 후 배포

## 5) 저장소 분리

- Hot storage: FAISS 인덱스 + segment metadata
- Warm storage: 재학습용 feature dump
- Cold storage: 원본 오디오(필요 시, 접근 통제)

검색 API는 Hot storage만 사용하도록 분리하면 비용/보안/운영 복잡도를 줄일 수 있습니다.

## 6) 운영 리스크 체크리스트

- 라이선스 증적: 소스별 계약서/약관 버전 보관
- 삭제 요청 대응: track_id 기반 인덱스 삭제 파이프라인
- 중복/커버곡 처리: 동일곡 다버전 매핑 정책
- 지역 제한: 국가별 사용 가능 카탈로그 분리

## 7) 현실적인 단계별 실행안

1. Stage A (2~4주): 오픈 데이터셋만으로 MVP 품질 검증
2. Stage B (4~8주): 계약 카탈로그 연동 + 증분 인덱싱
3. Stage C: 저작권/삭제 요청/감사 로그까지 포함한 운영 자동화

## 8) 참고 링크

- FMA Dataset README (license notes): https://github.com/mdeff/fma
- MTG-Jamendo Dataset: https://mtg.github.io/mtg-jamendo-dataset/
- Jamendo API Terms of Use: https://devportal.jamendo.com/api_terms_of_use
- AcoustID Web Service Guidelines: https://acoustid.org/webservice
- Spotify Developer Policy: https://developer.spotify.com/policy
- YouTube Developer Policies: https://developers.google.com/youtube/terms/developer-policies
