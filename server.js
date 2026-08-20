const path = require('path');
const http = require('http');
const express = require('express');
const { Server } = require('socket.io');

const PORT = process.env.PORT || 3001;
const ROUND_SECONDS = 60;
const prompts = ['太阳镜', '小猫', '火箭', '冰淇淋', '雨伞', '恐龙', '自行车', '汉堡', '城堡', '企鹅', '机器人', '彩虹', '向日葵', '热气球', '魔法棒', '鲸鱼'];
const rooms = new Map();
const app = express();
const server = http.createServer(app);
const io = new Server(server);
app.use(express.static(path.join(__dirname, 'public')));

function code() { const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; let value; do value = Array.from({ length: 4 }, () => chars[Math.floor(Math.random() * chars.length)]).join(''); while (rooms.has(value)); return value; }
function freshRoom(id) { return { id, players: new Map(), drawerIndex: 0, round: 0, prompt: '', strokes: [], guesses: [], scores: [0, 0], started: false, timeLeft: ROUND_SECONDS, timer: null, message: '' }; }
function publicState(room) { return { id: room.id, players: [...room.players].map(([socketId, player]) => ({ socketId, name: player.name, index: player.index, online: io.sockets.sockets.has(socketId) })), drawerIndex: room.drawerIndex, round: room.round, strokes: room.strokes, guesses: room.guesses.slice(-8), scores: room.scores, started: room.started, timeLeft: room.timeLeft, message: room.message }; }
function sendState(room) { io.to(room.id).emit('state', publicState(room)); for (const [socketId, player] of room.players) io.to(socketId).emit('privatePrompt', player.index === room.drawerIndex && room.started ? room.prompt : ''); }
function choosePrompt() { return prompts[Math.floor(Math.random() * prompts.length)]; }
function beginRound(room) {
  room.round += 1; room.prompt = choosePrompt(); room.strokes = []; room.guesses = []; room.timeLeft = ROUND_SECONDS; room.started = true; room.message = '';
  clearInterval(room.timer);
  room.timer = setInterval(() => { room.timeLeft -= 1; if (room.timeLeft <= 0) { room.timeLeft = 0; room.message = `时间到！答案是：${room.prompt}`; room.started = false; clearInterval(room.timer); setTimeout(() => { if (rooms.get(room.id) === room) beginRound(room); }, 2600); } sendState(room); }, 1000);
  sendState(room);
}
function playerFor(room, socket) { return room.players.get(socket.id); }
function roomFor(socket) { const id = [...socket.rooms].find((roomId) => roomId !== socket.id); return id && rooms.get(id); }

io.on('connection', (socket) => {
  socket.on('createRoom', ({ name } = {}, cb = () => {}) => { const id = code(); const room = freshRoom(id); room.players.set(socket.id, { name: String(name || '玩家 1').slice(0, 12), index: 0 }); rooms.set(id, room); socket.join(id); cb({ ok: true, id, index: 0 }); sendState(room); });
  socket.on('joinRoom', ({ id, name } = {}, cb = () => {}) => { const room = rooms.get(String(id || '').toUpperCase()); if (!room) return cb({ ok: false, message: '房间不存在。' }); if (room.players.size >= 2) return cb({ ok: false, message: '房间已满。' }); room.players.set(socket.id, { name: String(name || '玩家 2').slice(0, 12), index: 1 }); socket.join(room.id); cb({ ok: true, id: room.id, index: 1 }); if (room.players.size === 2 && !room.started) beginRound(room); else sendState(room); });
  socket.on('start', () => { const room = roomFor(socket); if (room && room.players.size === 2 && !room.started) beginRound(room); });
  socket.on('stroke', (stroke) => { const room = roomFor(socket); const player = room && playerFor(room, socket); if (!room || !room.started || !player || player.index !== room.drawerIndex) return; if (!stroke || !Array.isArray(stroke.points) || stroke.points.length < 2) return; room.strokes.push({ color: stroke.color === '#ef8354' ? '#ef8354' : '#2c3340', size: Math.min(24, Math.max(2, Number(stroke.size) || 6)), points: stroke.points.slice(0, 180) }); socket.to(room.id).emit('stroke', room.strokes.at(-1)); });
  socket.on('clearCanvas', () => { const room = roomFor(socket); const player = room && playerFor(room, socket); if (!room || !player || player.index !== room.drawerIndex) return; room.strokes = []; io.to(room.id).emit('canvasCleared'); });
  socket.on('guess', (text, cb = () => {}) => { const room = roomFor(socket); const player = room && playerFor(room, socket); const guess = String(text || '').trim().slice(0, 40); if (!room || !room.started || !player || player.index === room.drawerIndex || !guess) return cb({ ok: false, message: '现在不能提交这个答案。' }); const correct = guess.replace(/\s/g, '').toLowerCase() === room.prompt.replace(/\s/g, '').toLowerCase(); room.guesses.push({ name: player.name, text: guess, correct }); if (correct) { room.scores[player.index] += 1; room.scores[room.drawerIndex] += 1; room.message = `${player.name} 猜中了！答案是：${room.prompt}`; room.started = false; clearInterval(room.timer); setTimeout(() => { if (rooms.get(room.id) === room) { room.drawerIndex = room.drawerIndex === 0 ? 1 : 0; beginRound(room); } }, 2600); } sendState(room); cb({ ok: true, correct }); });
  socket.on('nextRound', () => { const room = roomFor(socket); const player = room && playerFor(room, socket); if (room && player && player.index === room.drawerIndex && !room.started) beginRound(room); });
  socket.on('disconnect', () => { const room = [...rooms.values()].find((item) => item.players.has(socket.id)); if (!room) return; room.message = '对方已离线，等待重新连接。'; sendState(room); if ([...room.players].every(([id]) => !io.sockets.sockets.has(id))) { clearInterval(room.timer); rooms.delete(room.id); } });
});

server.listen(PORT, () => console.log(`Draw & Guess running at http://localhost:${PORT}`));
