// DermaAI backend — Express + lowdb + Google Gemini API
// Setup: npm install  →  rename .env.example to .env and add your key  →  npm start
require('dotenv').config();
const express = require('express');
const cookieParser = require('cookie-parser');
const { v4: uuid } = require('uuid');
const low = require('lowdb');
const FileSync = require('lowdb/adapters/FileSync');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
app.use(express.json({ limit: '15mb' }));
app.use(cookieParser());
app.use(express.static(__dirname)); // serves index.html, css, js

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
// Models are tried in order. If one is overloaded (503/429) or retired (404),
// the server automatically moves on to the next one.
const MODEL_NAMES = ['gemini-3.8-flash', 'gemini-3.6-flash', 'gemini-3.5-flash-lite', 'gemini-2.5-flash'];
const models = MODEL_NAMES.map(name => ({ name, model: genAI.getGenerativeModel({ model: name }) }));
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function generateWithRetry(input) {
  let lastError;
  for (const { name, model } of models) {
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        return await model.generateContent(input);
      } catch (e) {
        lastError = e;
        if (e.status === 404) { console.log(`${name} not available, skipping`); break; }
        if (e.status === 503 || e.status === 429) {
          console.log(`${name} busy (${e.status}), attempt ${attempt}/2`);
          if (attempt < 2) await sleep(2000);
          continue;
        }
        throw e; // other errors (bad key, bad request) won't be fixed by retrying
      }
    }
    console.log(`Switching away from ${name}`);
  }
  throw lastError;
}

// ---------- database ----------
// Pure-JS file-based database (no native compiler needed) — stores everything
// in dermaai-db.json next to this file.
const adapter = new FileSync('dermaai-db.json');
const db = low(adapter);
db.defaults({ scans: [] }).write();

// ---------- sessions (anonymous, cookie-based) ----------
app.use((req, res, next) => {
  let sid = req.cookies.dermaai_sid;
  if (!sid) {
    sid = uuid();
    res.cookie('dermaai_sid', sid, { httpOnly: true, maxAge: 1000 * 60 * 60 * 24 * 365, sameSite: 'lax' });
  }
  req.sessionId = sid;
  next();
});

// ---------- shared system instructions ----------
const ANALYZE_INSTRUCTIONS = `You are a cautious skin & hair information assistant on a consumer website called DermaAI. You are NOT a doctor and must never state a diagnosis with certainty. Given an optional photo and/or a short user description, respond ONLY with JSON (no markdown fences) in this exact shape:
{"title": "short 3-6 word general topic label",
"urgent": boolean (true only if something could plausibly need prompt medical attention),
"body": "2-4 sentences, plain hedged language, never claim certainty",
"tips": ["3-5 short general self-care tips, each under 10 words"],
"seePro": "1 sentence on when/why to see a dermatologist or trichologist",
"category": "one of exactly: acne, dryness, eczema, dandruff, hairloss, oiliness, pigmentation, general"}
If the image is unclear or not actually skin/hair, say so honestly in "body" and use category "general".`;

function safeParseJSON(text) {
  try { return JSON.parse(text.replace(/```json|```/g, '').trim()); }
  catch { return null; }
}

// ---------- POST /api/analyze ----------
// body: { text?: string, imageBase64?: string }
app.post('/api/analyze', async (req, res) => {
  try {
    const { text, imageBase64 } = req.body;
    const parts = [{ text: ANALYZE_INSTRUCTIONS + '\n\n' + (text ? `User note: ${text}` : '(photo only, no note provided)') }];
    if (imageBase64) {
      parts.push({ inlineData: { mimeType: 'image/jpeg', data: imageBase64 } });
    }

    const result = await generateWithRetry(parts);
    const raw = result.response.text();
    const data = safeParseJSON(raw) || { title: 'Analysis unavailable', urgent: false, body: raw, tips: [], seePro: '', category: 'general' };

    // save to history
    const id = 's' + Date.now();
    const thumb = imageBase64 ? `data:image/jpeg;base64,${imageBase64.slice(0, 200000)}` : null; // small cap
    db.get('scans').push({
      id, session_id: req.sessionId,
      title: data.title, urgent: !!data.urgent, category: data.category,
      body: data.body, tips: data.tips || [], see_pro: data.seePro,
      thumb, created_at: Date.now()
    }).write();

    res.json({ ...data, id });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Analysis failed. Check server logs.' });
  }
});

// ---------- GET /api/history ----------
app.get('/api/history', (req, res) => {
  const rows = db.get('scans')
    .filter({ session_id: req.sessionId })
    .orderBy('created_at', 'desc')
    .take(20)
    .value();
  res.json(rows.map(r => ({
    id: r.id, title: r.title, urgent: !!r.urgent, category: r.category,
    body: r.body, tips: r.tips || [], seePro: r.see_pro,
    thumb: r.thumb, createdAt: r.created_at
  })));
});

// ---------- POST /api/chat ----------
// body: { context: {title, body}, history: [{role, content}], question: string }
app.post('/api/chat', async (req, res) => {
  try {
    const { context, history = [], question } = req.body;
    const sys = `You are DermaAI's follow-up assistant. Earlier you gave this general, non-diagnostic info — title: "${context.title}"; summary: "${context.body}". You are not a doctor. Answer the user's follow-up briefly (2-4 sentences), staying general and hedged, and recommend seeing a dermatologist/trichologist if relevant. Never claim a diagnosis.`;

    // Gemini's generateContent is single-turn here; fold the conversation into one prompt.
    const convoText = history.map(h => `${h.role === 'user' ? 'User' : 'Assistant'}: ${h.content}`).join('\n');
    const prompt = sys + (convoText ? `\n\nConversation so far:\n${convoText}` : '') + `\n\nUser: ${question}\n\nReply as the assistant, plain text only.`;

    const result = await generateWithRetry(prompt);
    res.json({ text: result.response.text() });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Chat failed.' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`DermaAI backend running → http://localhost:${PORT}`));
