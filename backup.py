import subprocess
import os
import random
from collections import Counter, defaultdict
from msoffcrypto import OfficeFile
import zipfile
import openpyxl
import tensorflow as tf  # TF 호환 모드 추가

# TF1 호환 모드 강제 활성화 (TF2 환경에서 TF1 코드 호환)
tf.compat.v1.disable_eager_execution()

# 1. GitHub 저장소 자동 클론 또는 업데이트 함수
def clone_or_update_repo(repo_url, local_path):
    if not os.path.exists(local_path):
        print(f"저장소 클론 중: {repo_url}")
        subprocess.run(["git", "clone", repo_url, local_path], check=True)
    else:
        print(f"저장소 업데이트 중: {local_path}")
        subprocess.run(["git", "-C", local_path, "pull"], check=True)

# 2. 방향 그래프 클래스 + 사이클 탐지
class DirectedGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = set()

    def add_node(self, node):
        # 노드 추가: 그래프에 새 문자 노드 삽입
        if node not in self.nodes:
            self.nodes.add(node)

    def add_edge(self, u, v, weight=1.0):
        # 에지 추가: u -> v 방향 에지, weight=확률 (중복 피함)
        self.add_node(u)
        self.add_node(v)
        if (v, weight) not in self.graph[u]:
            self.graph[u].append((v, weight))

    def remove_edge(self, u, v):
        self.graph[u] = [(nei, w) for nei, w in self.graph[u] if nei != v]

    def has_cycle(self, node, visited, rec_stack):
        visited[node] = True
        rec_stack[node] = True
        for neighbor, _ in self.graph[node]:
            if not visited.get(neighbor, False):
                if self.has_cycle(neighbor, visited, rec_stack):
                    return True
            elif rec_stack.get(neighbor, False):
                return True
        rec_stack[node] = False
        return False

    def detect_cycle(self):
        # 전체 그래프 사이클 검사: 모든 노드 순회
        visited = {node: False for node in self.nodes}
        rec_stack = {node: False for node in self.nodes}
        for node in list(self.nodes):
            if not visited[node]:
                if self.has_cycle(node, visited, rec_stack):
                    return True, visited, rec_stack
        return False, visited, rec_stack

# 3. PassGAN 후보 생성 함수 (사용자 학습 체크포인트 사용, TF 호환 + PYTHONPATH 추가)
def generate_candidates(passgan_dir, num_samples=50):
    output_file = os.path.join(passgan_dir, "gen_passwords.txt")
    # 사용자 학습 체크포인트 경로로 업데이트 (Colab rockyou 학습 모델: 195000.ckpt 베이스 이름)
    checkpoint_path = os.path.join(passgan_dir, "pretrained/checkpoints/195000.ckpt")

    # TF 호환 모드와 PYTHONPATH를 환경변수로 subprocess에 전달 (tflib 모듈 찾기)
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{passgan_dir}:{env.get('PYTHONPATH', '')}"  # tflib 경로 추가
    # TF 관련 환경변수 (호환성 강화, 로그 최소화)
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 에러만 출력 (deprecation/NCHW 숨김)

    result = subprocess.run([
        "python3",
        os.path.join(passgan_dir, "sample.py"),
        "--input-dir", os.path.join(passgan_dir, "pretrained"),
        "--checkpoint", checkpoint_path,
        "--output", output_file,
        "--batch-size", "64",  # 효율성 위해 조정 (M1 메모리 충분)
        "--num-samples", str(num_samples),
        "--seq-length", "10",  # 학습과 맞춤 (rockyou 평균 길이)
        "--layer-dim", "128"   # 학습과 맞춤 (LSTM 차원)
    ], env=env,  # 환경변수 전달
    capture_output=True, text=True)

    # 디버깅: subprocess 로그 출력 (restore SUCCESS 등 확인)
    if result.stdout:
        print("PassGAN 상세 로그:", result.stdout.strip())

    if result.returncode != 0:
        print("PassGAN 실행 오류:", result.stderr)
        return []

    candidates = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                pw = line.strip()
                if 5 <= len(pw) <= 8 and pw.isalnum():  # 필터링 조건 유지 (특수 문자 제외)
                    if pw not in candidates:  # 중복 방지 강화
                        candidates.append(pw)
        os.remove(output_file)
        print(f"PassGAN 후보 생성 완료: {len(candidates)}개 (필터링 후)")
    else:
        print("출력 파일 생성 실패: gen_passwords.txt 없음")
    return candidates[:num_samples]

# 4. 그래프 빌드 및 사이클 필터링 함수 (최적화: 중복 후보 제거 강화)
def build_and_filter_graph(candidates):
    G = DirectedGraph()
    charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    for char in charset:
        G.add_node(char)

    transitions = Counter()
    for pw in candidates:
        for i in range(len(pw) - 1):
            transitions[(pw[i], pw[i+1])] += 1

    total_trans = sum(transitions.values())
    for (u, v), count in transitions.items():
        weight = count / total_trans if total_trans > 0 else 1.0
        G.add_edge(u, v, weight)

    print(f"그래프: {len(G.nodes)} 노드, {sum(len(neis) for neis in G.graph.values())} 에지")

    has_cyc, visited, rec_stack = G.detect_cycle()
    removed = 0
    if has_cyc:
        # 사이클 에지 제거: rec_stack 기반 루프 에지 찾기 (반복 패턴 필터)
        for u in list(G.graph.keys()):
            for v, _ in G.graph[u][:]: 
                if rec_stack.get(v, False):
                    G.remove_edge(u, v)
                    removed += 1
    print(f"제거된 사이클 에지: {removed}")

    def dfs_weighted(start, length, path=[]):
        if len(path) == length:
            yield ''.join(path)
            return
        neighbors = G.graph.get(start, [])
        if neighbors:
            neighbors.sort(key=lambda x: x[1], reverse=True)
            for neighbor, _ in neighbors[:3]:
                path.append(neighbor)
                yield from dfs_weighted(neighbor, length, path)
                path.pop()

    safe_candidates = set()  # 중복 방지 위해 set 사용 (최적화)
    starts = random.sample(list(G.nodes), min(10, len(G.nodes)))
    for start in starts:
        for length in range(5, 9):
            for pw in list(dfs_weighted(start, length))[:5]:
                if pw not in safe_candidates:
                    safe_candidates.add(pw)
                    if len(safe_candidates) >= 30:
                        break
            if len(safe_candidates) >= 30:
                break
    return list(safe_candidates)  # list로 변환

# 5. 파일 크랙 함수 (기존 유지, 오류 상세 출력 추가)
def try_crack_file(file_path, password):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.zip':
            with zipfile.ZipFile(file_path) as zf:
                zf.setpassword(password.encode())
                if zf.testzip() is None:
                    print(f"ZIP 크랙 성공: {password}. 파일 목록: {zf.namelist()[:3]}...")
                    return True
        elif ext in ['.xlsx', '.xls']:
            with OfficeFile(file_path) as of:
                of.load_key(password=password)
                wb = openpyxl.load_workbook(of, password=password)
                print(f"Excel 크랙 성공: {password}. 시트: {wb.sheetnames}")
                return True
        elif ext == '.docx':
            with OfficeFile(file_path) as of:
                of.load_key(password=password)
                print(f"Word 크랙 성공: {password}.")
                return True
        elif ext == '.pptx':
            with OfficeFile(file_path) as of:
                of.load_key(password=password)
                print(f"PowerPoint 크랙 성공: {password}.")
                return True
        elif ext == '.txt':
            # TXT는 기본적으로 보호되지 않음, 하지만 가정 시도
            with open(file_path, 'r') as f:
                content = f.read(100)
                print(f"TXT 접근 성공: {password}. 미리보기: {content}...")
                return True
        else:
            print(f"지원 안 됨: {ext}")
            return False
    except Exception as e:
        print(f"크랙 시도 실패 ({password}): {str(e)}")  # 오류 상세 출력 추가
        return False

# 6. 메인 실행 함수 (기존 유지, 오류 처리 강화 + 현재 디렉터리 맞춤)
def crack_file_password(file_path, repo_url, local_repo_dir, max_retries=5):
    clone_or_update_repo(repo_url, local_repo_dir)

    if not os.path.exists(file_path):
        print("파일 없음!")
        return None

    print(f"대상 파일: {file_path}")
    attempt = 0
    while attempt < max_retries:
        print(f"[시도 {attempt + 1}] PassGAN으로 후보 생성 중...")
        candidates = generate_candidates(local_repo_dir, 50)
        if not candidates:
            print("후보 생성 실패, 다음 시도 스킵.")
            attempt += 1
            continue
        print(f"{len(candidates)}개 후보 생성됨.")

        print("그래프 빌드 및 사이클 필터링 중...")
        safe_candidates = build_and_filter_graph(candidates)
        print(f"{len(safe_candidates)}개 안전 후보 필터링 완료.")

        for pw in safe_candidates:
            if try_crack_file(file_path, pw):
                print(f"최종 크랙 성공: {pw}")
                return pw

        print("이번 시도 실패, 새로운 후보 생성 후 재시도 합니다.")
        attempt += 1

    print("최대 시도 횟수 초과, 크랙 실패.")
    return None

if __name__ == "__main__":
    file_path = input("크랙할 파일 경로 입력 (예: /path/to/file.zip): ").strip()
    repo_url = "https://github.com/MrLEESM/SoftwareProject.git"  # 또는 beta6/PassGAN으로 변경 가능
    local_repo_dir = "./PassGAN"  # 현재 디렉터리 맞춤 (이미 존재 시 pull)

    crack_file_password(file_path, repo_url, local_repo_dir)

# GUI (CLI 기반, GUI 필요 시 tkinter/streamlit 추가 가능)
