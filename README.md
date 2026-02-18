# find-song-in-the-fog

흥얼거리는 오디오(허밍)만으로 비슷한 곡을 찾는 실험용 프로젝트입니다.

현재는 두 가지 검색 경로를 제공합니다.

1. `legacy` 경로: FFT + 피치 겹침 기반(기존)
2. `ann` 경로: 멜로디 임베딩 + FAISS ANN + DTW 재순위화(신규)

대규모 환경 기준 추천은 `ann` 경로입니다. 원본 오디오 대신 검색용 표현(임베딩/지문)만 저장하도록 설계되어 있습니다.

## 프로젝트 구조

```text
find-song-in-the-fog/
  src/
    main.py               # CLI 엔트리포인트
    audio_analyzer.py     # 오디오 특성 추출(tempo, rms, chroma, FFT 시그니처)
    pitch_extractor.py    # YIN 기반 피치 추출, Hz -> MIDI -> 음이름 변환
    fft_matcher.py        # 코사인 유사도 + 피치 겹침 기반 결합 점수
    melody_embedding.py   # 멜로디 contour 추출 + 고정길이 임베딩
    ann_index.py          # FAISS(IVFPQ/HNSW/Flat) 인덱스 빌드/로드/검색
    dtw_reranker.py       # DTW + 반음 전이(키 보정) 재점수화
  data/
    index.json            # legacy 인덱스
    ann_index/            # ANN 아티팩트(melody.faiss, metadata.json, config)
  requirements.txt
  docs/
    data_collection.md    # 음원 수집/적재 전략
```

## ANN 경로 핵심 아이디어

1. 곡별로 여러 구간에서 멜로디 contour를 추출하고 정규화(키 중심 제거)
2. contour를 고정 길이 임베딩으로 변환해 FAISS 인덱스에 저장
3. 질의 허밍도 같은 방식으로 임베딩 생성 후 ANN Top-K 후보 조회
4. 후보 구간만 DTW로 정밀 재정렬해 최종 점수 산출

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`faiss-cpu` 설치가 실패하면 Python/OS 조합을 확인하거나 conda 환경에서 설치를 권장합니다.

## 사용법

### 1) ANN 인덱스 구축(권장)

```bash
python src/main.py index-ann --songs path/to/song_library --out-dir data/ann_index
```

주요 옵션:

- `--ann-type ivfpq|hnsw|flat` (기본: `ivfpq`)
- `--segment-seconds`, `--segment-hop-seconds`
- `--embedding-dim` (실제 벡터 차원은 2배: contour + delta)
- `--nlist`, `--m`, `--nbits`, `--nprobe` (IVFPQ 튜닝)

### 2) 허밍 질의 검색(ANN + DTW)

```bash
python src/main.py query-ann --hum path/to/hum.wav --index-dir data/ann_index --top-k 5
```

`--min-score` 이하 결과는 자동으로 제외되어 "유사곡 없음" 처리가 가능합니다.

출력 예시:

```text
Top matches (ANN + DTW)
 1. path/to/songA.mp3 | total=0.8421 (ann=0.9010, dtw=0.7698, shift=+2, seg=42.0-54.0s)
 2. path/to/songB.mp3 | total=0.8015 (ann=0.8550, dtw=0.7360, shift=-1, seg=30.0-42.0s)
```

### 3) 기존 방식(legacy)

```bash
python src/main.py index --songs path/to/song_library --out data/index.json
python src/main.py query --hum path/to/hum.wav --index data/index.json --top-k 5
```

## 운영 튜닝 힌트

- 정확도 부족: `candidate-k` 증가, `nprobe` 증가, `ann-weight` 낮춰 DTW 비중 확대
- 지연시간 과다: `candidate-k` 축소, DTW band(`--dtw-window-ratio`) 축소
- 메모리 부담: `ivfpq` + 작은 코드(`m`, `nbits`) 사용, 곡당 segment 수 제한

## 음원 데이터 수집 전략

실서비스용 수집/적재 전략은 `docs/data_collection.md`에 정리했습니다.
