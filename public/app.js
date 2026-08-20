const socket = io();
let myIndex = null, roomId = null, drawing = false, currentColor = '#2c3340', privatePrompt = '';
const lobby = document.querySelector('#lobby'), game = document.querySelector('#game'), canvas = document.querySelector('#canvas'), ctx = canvas.getContext('2d');
const toast = (text) => { const el = document.querySelector('#toast'); el.textContent = text; el.classList.add('show'); clearTimeout(toast.timer); toast.timer = setTimeout(() => el.classList.remove('show'), 2300); };
function enter(data) { myIndex = data.index; roomId = data.id; document.querySelector('#roomCode').textContent = roomId; lobby.classList.add('hidden'); game.classList.remove('hidden'); }
function clearLocal() { ctx.clearRect(0, 0, canvas.width, canvas.height); }
function renderState(state) {
  if (roomId === null) enter({ id: state.id, index: myIndex });
  document.querySelector('#roundLabel').textContent = state.started ? `第 ${state.round} 轮` : state.players.length < 2 ? '等待玩家' : '准备下一轮';
  const drawer = state.players.find((p) => p.index === state.drawerIndex); const me = state.players.find((p) => p.index === myIndex);
  const amDrawer = myIndex === state.drawerIndex;
  document.querySelector('#roleLabel').textContent = state.players.length < 2 ? '' : amDrawer ? '你来画' : '你来猜';
  document.querySelector('#promptBox').textContent = state.players.length < 2 ? '等待另一位玩家加入…' : state.started ? (amDrawer ? `请画：${privatePrompt || '提示加载中…'}` : '看画面，猜猜他画的是什么') : (state.message || '下一轮马上开始…');
  document.querySelector('#promptBox').classList.toggle('drawer', amDrawer);
  document.querySelector('#timer').textContent = state.timeLeft; document.querySelector('#timer').classList.toggle('warning', state.timeLeft <= 10);
  document.querySelector('#player0Name').textContent = state.players.find((p) => p.index === 0)?.name || '玩家 1'; document.querySelector('#player1Name').textContent = state.players.find((p) => p.index === 1)?.name || '等待加入';
  document.querySelector('#score0').textContent = state.scores[0]; document.querySelector('#score1').textContent = state.scores[1];
  document.querySelector('#guessInput').disabled = !state.started || amDrawer || state.players.length < 2;
  document.querySelector('#clearBtn').disabled = !state.started || !amDrawer;
  document.querySelector('#guessList').innerHTML = state.guesses.map((g) => `<div class="guess-item ${g.correct ? 'correct' : ''}"><b>${escapeHtml(g.name)}</b>${escapeHtml(g.text)}${g.correct ? ' ✓' : ''}</div>`).join('');
  if (state.message) document.querySelector('#message').textContent = state.message; else document.querySelector('#message').textContent = amDrawer && state.started ? '画得越像，越容易被猜中！' : state.started ? '轮到你猜了，输入答案试试。' : '';
}
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (c) => ({ '&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;' }[c])); }
function point(e) { const rect = canvas.getBoundingClientRect(); return { x: Math.round((e.clientX - rect.left) * canvas.width / rect.width), y: Math.round((e.clientY - rect.top) * canvas.height / rect.height) }; }
function drawStroke(stroke) { ctx.strokeStyle = stroke.color; ctx.lineWidth = stroke.size; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.beginPath(); stroke.points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.stroke(); }
canvas.addEventListener('pointerdown', (e) => { if (document.querySelector('#clearBtn').disabled) return; drawing = true; canvas.setPointerCapture(e.pointerId); canvas._points = [point(e)]; });
canvas.addEventListener('pointermove', (e) => { if (!drawing) return; const p = point(e); canvas._points.push(p); if (canvas._points.length > 1) drawStroke({ points: canvas._points.slice(-2), color: currentColor, size: Number(document.querySelector('#sizeInput').value) }); });
canvas.addEventListener('pointerup', () => { if (!drawing) return; drawing = false; socket.emit('stroke', { points: canvas._points, color: currentColor, size: Number(document.querySelector('#sizeInput').value) }); });
document.querySelector('#createBtn').onclick = () => socket.emit('createRoom', { name: document.querySelector('#nameInput').value }, (r) => { if (r.ok) { enter(r); toast(`房间 ${r.id} 已创建`); } });
document.querySelector('#joinBtn').onclick = () => socket.emit('joinRoom', { id: document.querySelector('#roomInput').value, name: document.querySelector('#nameInput').value }, (r) => { if (r.ok) enter(r); else document.querySelector('#lobbyMessage').textContent = r.message; });
document.querySelector('#roomInput').oninput = (e) => { e.target.value = e.target.value.toUpperCase().replace(/[^A-Z2-9]/g, ''); };
document.querySelector('#copyBtn').onclick = async () => { await navigator.clipboard?.writeText(roomId); toast('房间码已复制'); };
document.querySelector('#clearBtn').onclick = () => { clearLocal(); socket.emit('clearCanvas'); };
document.querySelectorAll('.color').forEach((button) => button.onclick = () => { currentColor = button.dataset.color; document.querySelectorAll('.color').forEach((b) => b.classList.remove('active')); button.classList.add('active'); });
document.querySelector('#guessForm').onsubmit = (e) => { e.preventDefault(); const input = document.querySelector('#guessInput'); if (!input.value.trim()) return; socket.emit('guess', input.value, (r) => { if (r.ok && r.correct) toast('猜中了！'); else if (!r.ok) toast(r.message); }); input.value = ''; };
socket.on('privatePrompt', (prompt) => { privatePrompt = prompt; }); socket.on('state', renderState); socket.on('stroke', drawStroke); socket.on('canvasCleared', clearLocal); socket.on('disconnect', () => { if (!game.classList.contains('hidden')) toast('网络连接已断开'); });
