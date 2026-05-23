import os
import uuid
import threading
import time

from flask import Flask, jsonify, request

import config
from environment import TicTacToeEnv
from agent import QLearningAgent
from train import run_episode

app = Flask(__name__)

# 전역 에이전트
agent_o: QLearningAgent | None = None
agent_x: QLearningAgent | None = None
agents_ready = False

train_status = {
    "running": False,
    "episode": 0,
    "total": 0,
    "draw_rate": 0.0,
    "done": False,
}

# 게임 세션 (메모리 내 관리)
sessions: dict[str, dict] = {}


# =============================================================================
# 시작 시 모델 로딩 / 학습
# =============================================================================

QUICK_EPISODES = int(os.environ.get("TRAIN_EPISODES", "100000"))
_UPDATE_INTERVAL = max(1, QUICK_EPISODES // 200)  # 200번 업데이트


def _quick_train():
    """모델이 없을 때 백그라운드에서 학습합니다."""
    global agent_o, agent_x, agents_ready

    train_status["running"] = True
    train_status["total"] = QUICK_EPISODES
    train_status["episode"] = 0

    env = TicTacToeEnv()
    ao = QLearningAgent("O")
    ax = QLearningAgent("X")

    for ep in range(1, QUICK_EPISODES + 1):
        run_episode(env, ao, ax)
        ao.decay_epsilon()
        ax.decay_epsilon()

        if ep % _UPDATE_INTERVAL == 0:
            train_status["episode"] = ep

    ao.save(config.MODEL_PATH_O)
    ax.save(config.MODEL_PATH_X)

    agent_o = ao
    agent_x = ax
    agents_ready = True
    train_status["running"] = False
    train_status["done"] = True
    train_status["episode"] = QUICK_EPISODES


def _load_models():
    """저장된 모델을 불러옵니다."""
    global agent_o, agent_x, agents_ready

    ao = QLearningAgent("O")
    ax = QLearningAgent("X")
    ao.load(config.MODEL_PATH_O)
    ax.load(config.MODEL_PATH_X)
    ao.epsilon = 0.0
    ax.epsilon = 0.0

    agent_o = ao
    agent_x = ax
    agents_ready = True
    train_status["done"] = True


def _init_agents():
    if os.path.exists(config.MODEL_PATH_O) and os.path.exists(config.MODEL_PATH_X):
        _load_models()
    else:
        _quick_train()


threading.Thread(target=_init_agents, daemon=True).start()


# =============================================================================
# REST API
# =============================================================================

@app.route("/api/train_status")
def api_train_status():
    return jsonify({
        "ready": agents_ready,
        "running": train_status["running"],
        "episode": train_status["episode"],
        "total": train_status["total"],
        "done": train_status["done"],
    })


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    """새 게임 세션을 생성합니다. robot_side: 'O' 또는 'X'"""
    if not agents_ready:
        return jsonify({"error": "학습 중입니다. 잠시 후 다시 시도하세요."}), 503

    data = request.get_json(silent=True) or {}
    robot_side = data.get("robot_side", "X").upper()
    if robot_side not in ("O", "X"):
        robot_side = "X"

    game_id = str(uuid.uuid4())
    env = TicTacToeEnv()
    state = env.reset()

    sessions[game_id] = {
        "env": env,
        "state": state,
        "robot_side": robot_side,
        "over": False,
        "winner": None,
    }

    # 로봇이 O(선공)이면 먼저 둡니다.
    msg = None
    if robot_side == "O":
        sessions[game_id], msg = _robot_move(sessions[game_id])

    return jsonify({
        "game_id": game_id,
        "board": sessions[game_id]["env"].board.tolist(),
        "current_player": sessions[game_id]["env"].current_player,
        "robot_side": robot_side,
        "over": sessions[game_id]["over"],
        "winner": sessions[game_id]["winner"],
        "message": msg,
        "valid_actions": sessions[game_id]["env"].get_valid_actions(),
    })


@app.route("/api/move", methods=["POST"])
def api_move():
    """사람이 수를 둡니다. game_id와 action(칸 번호 0~15)을 전달합니다."""
    data = request.get_json(silent=True) or {}
    game_id = data.get("game_id")
    action = data.get("action")

    if game_id not in sessions:
        return jsonify({"error": "게임을 찾을 수 없습니다."}), 404

    session = sessions[game_id]
    if session["over"]:
        return jsonify({"error": "이미 종료된 게임입니다."}), 400

    env: TicTacToeEnv = session["env"]

    if action not in env.get_valid_actions():
        return jsonify({"error": "유효하지 않은 위치입니다."}), 400

    # 사람 수 두기
    state, r_o, r_x, done = env.step(action)
    session["state"] = state

    message = None
    if done:
        _resolve_done(session, r_o, r_x)
    else:
        # 로봇 차례
        session, message = _robot_move(session)

    return jsonify({
        "board": env.board.tolist(),
        "current_player": env.current_player,
        "over": session["over"],
        "winner": session["winner"],
        "message": message,
        "valid_actions": env.get_valid_actions() if not session["over"] else [],
    })


# =============================================================================
# 내부 헬퍼
# =============================================================================

def _robot_move(session: dict) -> tuple[dict, str | None]:
    env: TicTacToeEnv = session["env"]
    valid = env.get_valid_actions()
    if not valid or env.done:
        return session, None

    agent = agent_o if env.current_player == 1 else agent_x
    action = agent.choose_action(session["state"], valid)
    state, r_o, r_x, done = env.step(action)
    session["state"] = state

    message = f"AI가 {action}번 칸에 두었습니다."
    if done:
        _resolve_done(session, r_o, r_x)

    return session, message


def _resolve_done(session: dict, r_o: float, r_x: float):
    session["over"] = True
    robot_side = session["robot_side"]
    if r_o == config.REWARD_WIN:
        session["winner"] = "O"
    elif r_x == config.REWARD_WIN:
        session["winner"] = "X"
    else:
        session["winner"] = "draw"


# =============================================================================
# 프론트엔드 HTML
# =============================================================================

HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>4×4 틱택토 AI</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }
  h1 { font-size: 1.8rem; margin-bottom: 6px; color: #38bdf8; }
  .subtitle { color: #94a3b8; margin-bottom: 24px; font-size: 0.9rem; }

  #status-bar {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 20px;
    margin-bottom: 20px;
    text-align: center;
    min-width: 320px;
  }

  .controls {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
    justify-content: center;
  }
  .controls label { color: #94a3b8; align-self: center; }
  select, button {
    padding: 8px 16px;
    border-radius: 6px;
    border: none;
    font-size: 0.95rem;
    cursor: pointer;
  }
  select { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; }
  button {
    background: #0284c7;
    color: white;
    font-weight: 600;
    transition: background 0.2s;
  }
  button:hover { background: #0369a1; }
  button:disabled { background: #334155; cursor: not-allowed; }

  #board {
    display: grid;
    grid-template-columns: repeat(4, 80px);
    grid-template-rows: repeat(4, 80px);
    gap: 4px;
    margin-bottom: 20px;
  }
  .cell {
    background: #1e293b;
    border: 2px solid #334155;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    font-weight: 700;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    position: relative;
  }
  .cell:hover:not(.taken):not(.disabled) {
    background: #334155;
    border-color: #38bdf8;
  }
  .cell.taken { cursor: default; }
  .cell.disabled { cursor: default; }
  .cell.O { color: #34d399; }
  .cell.X { color: #f87171; }
  .cell.hint { border-color: #38bdf8; background: #1a3347; }
  .cell-num {
    position: absolute;
    top: 4px;
    right: 6px;
    font-size: 0.55rem;
    color: #475569;
    font-weight: 400;
  }

  #message {
    min-height: 32px;
    text-align: center;
    color: #fbbf24;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 12px;
  }

  .result-O { color: #34d399 !important; }
  .result-X { color: #f87171 !important; }
  .result-draw { color: #94a3b8 !important; }

  #progress-wrap {
    width: 320px;
    background: #1e293b;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 12px;
  }
  #progress-bar {
    height: 8px;
    background: #0284c7;
    width: 0%;
    transition: width 0.5s;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .pulsing { animation: pulse 1.2s ease-in-out infinite; }
</style>
</head>
<body>
<h1>4×4 틱택토 AI</h1>
<p class="subtitle">Q-러닝으로 학습한 AI와 대전하세요</p>

<div id="status-bar">⏳ AI 준비 중...</div>

<div id="progress-wrap" style="display:none">
  <div id="progress-bar"></div>
</div>

<div class="controls">
  <label>내가 두는 말:</label>
  <select id="human-side">
    <option value="X">O 먼저 (AI가 O)</option>
    <option value="O">X 나중 (AI가 X)</option>
  </select>
  <button id="btn-new" disabled onclick="newGame()">새 게임</button>
</div>

<div id="board"></div>
<div id="message"></div>

<script>
let gameId = null;
let myTurn = false;
let robotSide = 'X';
let boardReady = false;
let trainStart = null;

const boardEl = document.getElementById('board');
const msgEl = document.getElementById('message');
const statusEl = document.getElementById('status-bar');
const btnNew = document.getElementById('btn-new');
const progressWrap = document.getElementById('progress-wrap');
const progressBar = document.getElementById('progress-bar');

// ── 보드 초기화 ──────────────────────────────────────────────────────────
function initBoard() {
  boardEl.innerHTML = '';
  for (let i = 0; i < 16; i++) {
    const cell = document.createElement('div');
    cell.className = 'cell disabled';
    cell.dataset.idx = i;
    cell.innerHTML = `<span class="cell-num">${i}</span>`;
    cell.addEventListener('click', () => onCellClick(i));
    boardEl.appendChild(cell);
  }
}

function renderBoard(board, validActions) {
  const cells = boardEl.querySelectorAll('.cell');
  cells.forEach((cell, i) => {
    const val = board[i];
    const content = val === 1 ? 'O' : val === -1 ? 'X' : '';
    cell.className = 'cell' + (val !== 0 ? ' taken ' + content : '');
    cell.innerHTML = `<span class="cell-num">${i}</span>${content}`;
    if (myTurn && validActions && validActions.includes(i) && val === 0) {
      cell.classList.add('hint');
    }
    if (!myTurn || val !== 0) {
      cell.classList.add('disabled');
    }
  });
}

function onCellClick(idx) {
  if (!myTurn || !gameId) return;
  makeMove(idx);
}

// ── API 호출 ─────────────────────────────────────────────────────────────
function elapsedStr() {
  if (!trainStart) return '';
  const sec = Math.floor((Date.now() - trainStart) / 1000);
  const m = Math.floor(sec / 60), s = sec % 60;
  return m > 0 ? ` (${m}분 ${s}초 경과)` : ` (${s}초 경과)`;
}

async function checkReady() {
  try {
    const r = await fetch('/api/train_status');
    const d = await r.json();
    if (d.ready) {
      statusEl.textContent = '✅ AI 준비 완료 — 새 게임을 시작하세요';
      statusEl.classList.remove('pulsing');
      btnNew.disabled = false;
      progressWrap.style.display = 'none';
      boardReady = true;
    } else if (d.running) {
      if (!trainStart) trainStart = Date.now();
      statusEl.classList.add('pulsing');
      const pct = d.total > 0 ? Math.round(d.episode / d.total * 100) : 0;
      statusEl.textContent = `🧠 AI 학습 중... ${d.episode.toLocaleString()} / ${d.total.toLocaleString()} (${pct}%)${elapsedStr()}`;
      progressWrap.style.display = 'block';
      progressBar.style.width = pct + '%';
      setTimeout(checkReady, 1000);
    } else {
      statusEl.classList.add('pulsing');
      statusEl.textContent = '⏳ AI 초기화 중...';
      setTimeout(checkReady, 800);
    }
  } catch {
    setTimeout(checkReady, 2000);
  }
}

async function newGame() {
  robotSide = document.getElementById('human-side').value;
  const res = await fetch('/api/new_game', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({robot_side: robotSide})
  });
  const d = await res.json();
  if (d.error) { alert(d.error); return; }

  gameId = d.game_id;
  myTurn = !d.over && d.current_player !== (robotSide === 'O' ? 1 : -1);

  // 사람이 O면 current_player==1이 내 차례, X면 current_player==-1이 내 차례
  const humanPlayer = robotSide === 'O' ? -1 : 1;
  myTurn = !d.over && (d.current_player === humanPlayer);

  renderBoard(d.board, d.valid_actions);
  const sideLabel = robotSide === 'O' ? '당신은 X' : '당신은 O';
  msgEl.className = '';
  msgEl.textContent = d.message ? d.message : `게임 시작! ${sideLabel}`;
  if (d.over) finishGame(d.winner);
}

async function makeMove(action) {
  myTurn = false;
  const res = await fetch('/api/move', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({game_id: gameId, action})
  });
  const d = await res.json();
  if (d.error) { myTurn = true; alert(d.error); return; }

  const humanPlayer = robotSide === 'O' ? -1 : 1;
  myTurn = !d.over && (d.current_player === humanPlayer);

  renderBoard(d.board, d.valid_actions);
  msgEl.className = '';
  msgEl.textContent = d.message || '';

  if (d.over) finishGame(d.winner);
}

function finishGame(winner) {
  myTurn = false;
  const humanSide = robotSide === 'O' ? 'X' : 'O';
  if (winner === 'draw') {
    msgEl.className = 'result-draw';
    msgEl.textContent = '무승부! 🤝';
  } else if (winner === humanSide) {
    msgEl.className = 'result-' + humanSide;
    msgEl.textContent = '당신이 이겼습니다! 🎉';
  } else {
    msgEl.className = 'result-' + winner;
    msgEl.textContent = 'AI가 이겼습니다! 🤖';
  }
}

// ── 시작 ─────────────────────────────────────────────────────────────────
initBoard();
checkReady();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    from flask import Response
    return Response(HTML, mimetype="text/html")


# =============================================================================
# 서버 실행
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
