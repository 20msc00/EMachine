const grid = document.getElementById('grid');
const statusEl = document.getElementById('status');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const personaLabel = document.getElementById('persona-list');
const companionLabel = document.getElementById('companion-list');

const conversations = new Map();

function formatPersonaName(filename) {
  return filename
    .replace(/\.md$/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function ensureColumn(event) {
  if (conversations.has(event.id)) return conversations.get(event.id);
  const column = document.createElement('article');
  column.className = 'column';
  column.dataset.id = event.id;

  const head = document.createElement('header');
  const h2 = document.createElement('h2');
  h2.textContent = formatPersonaName(event.persona);
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = `${event.companion} · seed ${event.seed}`;
  head.appendChild(h2);
  head.appendChild(meta);

  const messages = document.createElement('div');
  messages.className = 'messages';

  const summary = document.createElement('div');
  summary.className = 'summary';
  summary.textContent = 'Waiting...';

  column.appendChild(head);
  column.appendChild(messages);
  column.appendChild(summary);
  grid.appendChild(column);

  const record = { column, messages, summary, persona: formatPersonaName(event.persona), companion: event.companion, tokens: { assistant: 0, user: 0 } };
  conversations.set(event.id, record);
  return record;
}

function addMessage(event) {
  const record = ensureColumn(event);
  const bubble = document.createElement('div');
  bubble.className = `message ${event.role}`;
  bubble.textContent = event.content;
  record.messages.appendChild(bubble);
  record.messages.scrollTop = record.messages.scrollHeight;
}

function updateSummary(id, info) {
  const record = conversations.get(id);
  if (!record) return;
  record.tokens.assistant = info.companion_tokens || record.tokens.assistant;
  record.tokens.user = info.persona_tokens || record.tokens.user;
  record.summary.innerHTML = `<span class="badge">${info.end_reason}</span> assistant ${record.tokens.assistant}t · user ${record.tokens.user}t`;
}

async function fetchConfig() {
  const res = await fetch('/config');
  if (!res.ok) return;
  const data = await res.json();
  personaLabel.textContent = data.personas.map(formatPersonaName).join(', ');
  companionLabel.textContent = data.companions.join(', ');
}

function connect() {
  const url = `${location.origin.replace('http', 'ws')}/ws`;
  const socket = new WebSocket(url);
  socket.addEventListener('open', () => {
    statusEl.textContent = 'Connected';
  });
  socket.addEventListener('close', () => {
    statusEl.textContent = 'Disconnected';
    setTimeout(connect, 1000);
  });
  socket.addEventListener('message', (evt) => {
    const payload = JSON.parse(evt.data);
    if (payload.type === 'run_start') {
      statusEl.textContent = `Running (${payload.total})`;
      grid.innerHTML = '';
      conversations.clear();
    } else if (payload.type === 'conversation_start') {
      ensureColumn(payload);
    } else if (payload.type === 'message') {
      addMessage(payload);
    } else if (payload.type === 'conversation_end') {
      updateSummary(payload.id, payload);
    } else if (payload.type === 'run_complete') {
      statusEl.textContent = 'Completed';
    } else if (payload.type === 'run_stopped') {
      statusEl.textContent = 'Stopped';
      conversations.forEach((record) => {
        if (record.summary && record.summary.textContent === 'Waiting...') {
          record.summary.textContent = 'Stopped';
        }
      });
    }
  });
}

startBtn.addEventListener('click', () => {
  fetch('/start', { method: 'POST' });
});

stopBtn.addEventListener('click', () => {
  fetch('/stop', { method: 'POST' });
});

fetchConfig();
connect();
