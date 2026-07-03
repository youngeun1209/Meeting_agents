# meeting agents

meeting agents는 회의 오디오를 실시간 STT로 받아 화면에 보여주고, 말하는 중에도 실시간 요약을 확인할 수 있으며, 전체 transcript 번역과 회의록 작성까지 해주는 로컬 회의 도우미입니다.

Zoom, Google Meet, 온라인 강의처럼 Mac에서 재생되는 소리와 내 마이크를 동시에 받아 `[Others]`, `[Me]`로 구분합니다. 기본 흐름은 로컬 실행이며, STT 결과는 자동으로 `transcripts/` 폴더에 저장됩니다.

English version: [README.md](README.md)

## 설치

먼저 프로젝트 폴더로 이동합니다.

```bash
cd Meeting_agents
```

오디오 입력에 필요한 PortAudio를 설치합니다.

```bash
brew install portaudio
```

Python 가상환경을 만들고 패키지를 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Apple Silicon Mac에서는 기본 STT 백엔드인 `mlx`를 쓰는 것이 빠릅니다.

```bash
pip install mlx-whisper
```

시스템 오디오 캡처용 네이티브 사이드카를 빌드합니다.

```bash
bash native/build.sh
```

이 단계는 macOS 13 이상, Xcode Command Line Tools의 `swiftc`가 필요합니다. 처음 실행할 때 macOS가 Screen Recording 권한을 물어보면 허용한 뒤 앱을 다시 실행하세요.

## 번역과 회의록 생성을 위한 Ollama 설치

STT만 사용할 경우 Ollama는 없어도 됩니다. 다만 transcript 번역과 회의록 생성 기능을 쓰려면 로컬 LLM 서버가 필요합니다.

```bash
brew install ollama
ollama serve
```

다른 터미널을 하나 더 열고 모델을 받습니다.

```bash
ollama pull qwen2.5:7b
```

가벼운 모델이 필요하면 `qwen2.5:3b`를 받을 수 있고, 사용할 모델 이름은 [config.py](config.py)의 `OLLAMA_MODEL`에서 바꿀 수 있습니다.

## 실행

GUI 실행이 기본 사용 방식입니다.

```bash
python gui.py
```

CLI로 터미널에서 바로 실행할 수도 있습니다.

```bash
python main.py
```

입력 장치 이름을 확인하려면 다음 명령을 사용합니다.

```bash
python main.py --list-devices
```

## 사용하는 방법

1. `python gui.py`로 앱을 실행합니다.
2. 상단에서 STT 모델과 언어를 선택합니다.
   - `Auto`: 한국어/영어/중국어 자동 감지
   - `Korean`: 한국어 고정
   - `English`: 영어 고정
   - `Chinese`: 중국어 고정
3. `Start`를 누르면 회의 오디오 캡처와 STT가 시작됩니다.
4. 회의를 진행하면 실시간 자막이 화면에 쌓이고, 각 발화는 자동으로 `.txt` 파일에 저장됩니다.
5. `Live Summary` 탭에서 발화자가 이어서 말한 내용을 묶어 실시간으로 볼 수 있습니다.
6. 회의가 끝나면 `Stop`을 누릅니다.
7. `Translator` 탭에서 Korean, English, Chinese 중 하나를 고르고 `Translate`를 누르면 전체 transcript를 번역합니다.
8. 생성된 번역문은 저장 버튼으로 `translation_*.md` 파일로 저장할 수 있습니다.
9. `Minutes` 탭에서 `Generate`를 누르면 Ollama가 전체 transcript를 읽고 회의록을 생성합니다.
10. 생성된 회의록은 저장 버튼으로 `minutes_*.md` 파일로 저장할 수 있습니다.

`New`는 현재 화면을 비우고 새 transcript 파일로 다음 회의를 시작할 때 사용합니다.

## 작동 메커니즘

meeting agents는 두 종류의 오디오 소스를 동시에 캡처합니다.

- `Others`: Mac에서 재생되는 시스템 오디오입니다. Zoom, Meet, 강의 영상, 브라우저 소리 등이 여기에 해당합니다.
- `Me`: 내 물리 마이크 입력입니다.

캡처된 오디오는 짧은 블록 단위로 큐에 들어가고, `VADBuffer`가 침묵 구간을 기준으로 발화 단위를 나눕니다. 나뉜 발화는 Whisper 계열 STT 모델로 전달되어 텍스트가 됩니다. 텍스트는 즉시 화면에 표시되고, 동시에 transcript 파일에 저장됩니다.

번역과 회의록 생성은 저장된 transcript 전체를 Ollama 로컬 LLM에 보내는 방식입니다. 외부 API를 사용하지 않으며, 기본 설정에서는 회의 내용이 로컬 머신 밖으로 나가지 않습니다.

## 설정 변경

주요 설정은 [config.py](config.py)에서 바꿉니다.

| 원하는 동작 | 변경할 값 |
| --- | --- |
| 더 빠르게 인식 | `MODEL_SIZE="base"` 또는 `"small"` |
| 더 정확하게 인식 | `MODEL_SIZE="medium"` 또는 `"large-v3-turbo"` |
| 한국어만 인식 | `LANGUAGE="ko"` |
| 영어만 인식 | `LANGUAGE="en"` |
| 중국어만 인식 | `LANGUAGE="zh"` |
| 자동 언어 감지 | `LANGUAGE=None` |
| 짧은 말이 자주 잘림 | `VAD_SILENCE_SEC`를 조금 올리기 |
| 조용한 소리를 놓침 | `MIN_RMS`를 낮추기 |
| 잡음이 텍스트로 나옴 | `MIN_RMS` 또는 `VAD_AGGRESSIVENESS` 올리기 |
| 회의록 모델 변경 | `OLLAMA_MODEL` 변경 |

마이크 장치 이름이 다르면 `python main.py --list-devices`로 확인한 뒤 [config.py](config.py)의 `SOURCES`에서 `device` 값을 맞춥니다.

```python
SOURCES = [
    {"kind": "system", "speaker": "Others"},
    {"device": "MacBook Pro Microphone", "speaker": "Me"},
]
```

상대방/강의 소리만 받고 내 마이크를 제외하려면 두 번째 줄을 지우면 됩니다.

## 생성되는 파일

실시간 STT 결과는 `transcripts/transcript_*.txt`로 저장됩니다. GUI에서 번역문이나 회의록을 저장하면 `translation_*.md`, `minutes_*.md` 형식의 Markdown 파일이 생성됩니다.

## 파일 설명

- [config.py](config.py): STT 모델, 언어, 오디오 소스, VAD, 저장 위치, Ollama 모델 설정
- [gui.py](gui.py): GUI 앱. 실시간 STT, 실시간 요약, 번역, 회의록 생성 화면을 제공
- [main.py](main.py): CLI 실행 진입점. 터미널에서 STT를 실행하거나 장치 목록을 확인
- [audio_capture.py](audio_capture.py): 마이크와 시스템 오디오를 캡처해 발화 처리 큐로 전달
- [vad_buffer.py](vad_buffer.py): 오디오 블록을 침묵 기준으로 발화 단위로 분리
- [transcriber.py](transcriber.py): Whisper 백엔드를 감싸 실제 STT를 수행
- [writer.py](writer.py): STT 결과를 화면/GUI로 전달하고 transcript 파일로 저장
- [minutes.py](minutes.py): Ollama 로컬 LLM을 호출해 transcript를 회의록으로 변환
- [translator.py](translator.py): Ollama 로컬 LLM을 호출해 transcript를 번역
- [native/sysaudio.swift](native/sysaudio.swift): macOS ScreenCaptureKit 기반 시스템 오디오 캡처 사이드카
- [native/build.sh](native/build.sh): 시스템 오디오 사이드카 빌드 스크립트
- [requirements.txt](requirements.txt): Python 의존성 목록
