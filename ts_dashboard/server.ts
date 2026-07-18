import express from "express";
import cors from "cors";
import path from "path";
import sqlite3 from "sqlite3";
import { WebSocketServer, WebSocket } from "ws";
import http from "http";

const app = express();
const port = 4001;
const dbPath = path.join(__dirname, "..", "shared", "metrics.db");

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));
app.use("/api/profile", express.static(path.join(__dirname, "..", "shared", "prof_logs")));

// Open SQLite database in READONLY mode
const openDatabase = (): sqlite3.Database => {
  return new sqlite3.Database(dbPath, sqlite3.OPEN_READONLY, (err) => {
    if (err) {
      console.error("[ERROR] Failed to connect to SQLite database:", err.message);
    } else {
      console.log("[INFO] Connected to SQLite metrics database in READONLY mode.");
    }
  });
};

const db = openDatabase();

// REST API: Get historical metrics by name (e.g., loss, tokens_per_sec, step_time_ms)
app.get("/api/metrics/:name", (req, res) => {
  const { name } = req.params;
  const since = Number(req.query.since || 0);

  const query = `
    SELECT step, value, timestamp, node_id 
    FROM metrics 
    WHERE name = ? AND step > ? 
    ORDER BY step ASC
  `;

  db.all(query, [name, since], (err, rows) => {
    if (err) {
      console.error("[ERROR] Database query failed:", err.message);
      res.status(500).json({ error: err.message });
      return;
    }
    res.json(rows);
  });
});

// REST API: Get aggregate summary metrics
app.get("/api/summary", (req, res) => {
  const query = `
    SELECT 
      (SELECT MAX(step) FROM metrics) as current_step,
      (SELECT value FROM metrics WHERE name = 'loss' ORDER BY step DESC LIMIT 1) as latest_loss,
      (SELECT AVG(value) FROM metrics WHERE name = 'loss' AND step > (SELECT MAX(step) - 100 FROM metrics)) as avg_loss_100,
      (SELECT value FROM metrics WHERE name = 'tokens_per_sec' ORDER BY step DESC LIMIT 1) as tokens_per_sec
  `;

  db.get(query, [], (err, row) => {
    if (err) {
      console.error("[ERROR] Summary query failed:", err.message);
      res.status(500).json({ error: err.message });
      return;
    }
    res.json(row || { current_step: 0, latest_loss: 0, avg_loss_100: 0, tokens_per_sec: 0 });
  });
});

// Fallback to serve index.html for any frontend routing
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// Create HTTP and WebSocket Server
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

import { spawn } from "child_process";

// WebSocket connection handler
wss.on("connection", (ws: WebSocket) => {
  console.log("[INFO] WebSocket client connected.");
  
  // Track the last step we sent to this client to prevent duplicate data
  let lastSentStep = -1;

  // Handle incoming messages (e.g. prompt chat requests)
  ws.on("message", (message: string) => {
    try {
      const payload = JSON.parse(message);
      if (payload.type === "chat_request") {
        const { prompt, checkpoint, temperature, top_p } = payload.data;
        console.log(`[INFO] Received chat request for prompt: "${prompt}" using checkpoint: ${checkpoint}`);
        
        let checkPath = checkpoint;
        if (!path.isAbsolute(checkpoint)) {
          checkPath = path.join(__dirname, "..", "shared", "checkpoints", checkpoint);
        }
        
        const postData = JSON.stringify({
          prompt,
          checkpoint: checkPath,
          temperature: temperature || 0.7,
          top_p: top_p || 0.9
        });

        const req = http.request({
          hostname: "localhost",
          port: 4002,
          path: "/generate",
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Content-Length": Buffer.byteLength(postData)
          }
        }, (res) => {
          res.on("data", (chunk) => {
            const text = chunk.toString();
            if (text) {
              ws.send(JSON.stringify({ type: "chat_token", data: text }));
            }
          });
          
          res.on("end", () => {
            ws.send(JSON.stringify({ type: "chat_end", data: { code: 0 } }));
          });
        });

        req.on("error", (e) => {
          console.error("[ERROR] Failed to communicate with Python inference server:", e.message);
          ws.send(JSON.stringify({ type: "chat_token", data: "\nHata: Model sunucusuna bağlanılamadı. Arka planda sunucunun başlatıldığından emin oluyorum..." }));
          ws.send(JSON.stringify({ type: "chat_end", data: { code: 1 } }));
        });

        req.write(postData);
        req.end();
      }
    } catch (e: any) {
      console.error("[ERROR] Failed to process WebSocket message:", e.message);
    }
  });

  // Poll database every 2 seconds and stream new metrics to client
  const intervalId = setInterval(() => {
    if (ws.readyState !== WebSocket.OPEN) return;

    const query = `
      SELECT step, name, value, node_id, timestamp 
      FROM metrics 
      WHERE step > ? 
      ORDER BY step ASC
    `;

    db.all(query, [lastSentStep], (err, rows: any[]) => {
      if (err) {
        console.error("[ERROR] WebSocket DB poll failed:", err.message);
        return;
      }

      if (rows && rows.length > 0) {
        // Broadcast the metrics batch
        ws.send(JSON.stringify({ type: "metrics", data: rows }));
        
        // Update the last sent step to the maximum step in this batch
        const maxStep = rows[rows.length - 1].step;
        if (maxStep > lastSentStep) {
          lastSentStep = maxStep;
        }
      }
    });
  }, 2000);

  ws.on("close", () => {
    console.log("[INFO] WebSocket client disconnected.");
    clearInterval(intervalId);
  });
});

server.listen(port, () => {
  console.log(`[INFO] TypeScript training dashboard backend running on http://localhost:${port}`);
});
