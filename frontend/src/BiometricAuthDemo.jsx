import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================
// BIOMETRIC AUTH DEMO — Self-contained React component
// Captures real keystrokes & mouse, runs local ML scoring,
// simulates adversarial attacks, and shows defense mechanisms
// ============================================================

// --- Inline ML: lightweight One-Class scoring ---
function gaussianScore(sample, mean, std) {
  let logProb = 0;
  for (let i = 0; i < sample.length; i++) {
    const s = std[i] || 0.001;
    const diff = sample[i] - mean[i];
    logProb -= (diff * diff) / (2 * s * s);
  }
  return logProb / sample.length;
}

function cosineSimilarity(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB) + 1e-8);
}

// --- Feature extraction from raw browser events ---
function extractKeystrokeFeatures(events) {
  if (events.length < 3) return null;
  const holds = [], dds = [], uds = [];
  for (const e of events) {
    if (e.up && e.down) holds.push((e.up - e.down) / 1000);
  }
  for (let i = 1; i < events.length; i++) {
    if (events[i].down && events[i - 1].down)
      dds.push((events[i].down - events[i - 1].down) / 1000);
    if (events[i].down && events[i - 1].up)
      uds.push((events[i].down - events[i - 1].up) / 1000);
  }
  const s = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
  const sd = (a) => {
    if (a.length < 2) return 0;
    const m = s(a);
    return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length);
  };
  return [
    ...holds.slice(0, 5),
    ...Array(Math.max(0, 5 - holds.length)).fill(0),
    ...dds.slice(0, 5),
    ...Array(Math.max(0, 5 - dds.length)).fill(0),
    s(holds), sd(holds), s(dds), sd(dds), s(uds), sd(uds),
    events.length / (dds.reduce((a, b) => a + b, 0) || 1),
    sd(holds) / (s(holds) || 1),
    holds.length > 0 ? Math.max(...holds) : 0,
    uds.filter((u) => u < 0).length / (uds.length || 1),
  ];
}

function extractMouseFeatures(trail) {
  if (trail.length < 5) return null;
  const speeds = [], angles = [];
  for (let i = 1; i < trail.length; i++) {
    const dx = trail[i].x - trail[i - 1].x;
    const dy = trail[i].y - trail[i - 1].y;
    const dt = Math.max((trail[i].t - trail[i - 1].t) / 1000, 0.001);
    speeds.push(Math.min(Math.sqrt(dx * dx + dy * dy) / dt, 5000));
    angles.push(Math.atan2(dy, dx));
  }
  const s = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const sd = (a) => {
    const m = s(a);
    return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length);
  };
  const totalDist = speeds.reduce((a, b) => a + b, 0) * 0.015;
  const disp = Math.sqrt(
    (trail[trail.length - 1].x - trail[0].x) ** 2 +
      (trail[trail.length - 1].y - trail[0].y) ** 2
  );
  return [
    s(speeds), sd(speeds), Math.max(...speeds),
    s(angles), sd(angles),
    totalDist, disp, disp / (totalDist + 1),
    (trail[trail.length - 1].t - trail[0].t) / 1000,
    angles.filter((_, i) => i > 0 && Math.abs(angles[i] - angles[i - 1]) > 0.5).length,
  ];
}

// --- FGSM Attack Simulation (local) ---
function fgsmAttack(sample, mean, std, epsilon) {
  const grad = sample.map((v, i) => {
    const s = std[i] || 0.001;
    return -(v - mean[i]) / (s * s);
  });
  return sample.map((v, i) => v + epsilon * Math.sign(grad[i]));
}

function pgdAttack(sample, mean, std, epsilon, steps) {
  let adv = [...sample];
  const alpha = epsilon / 4;
  const trajectory = [];
  for (let step = 0; step < steps; step++) {
    const grad = adv.map((v, i) => {
      const s = std[i] || 0.001;
      return -(v - mean[i]) / (s * s);
    });
    adv = adv.map((v, i) => {
      let newV = v + alpha * Math.sign(grad[i]);
      const delta = Math.max(-epsilon, Math.min(epsilon, newV - sample[i]));
      return sample[i] + delta;
    });
    trajectory.push(gaussianScore(adv, mean, std));
  }
  return { adversarial: adv, trajectory };
}

function mimicryAttack(mean, std, noise) {
  return mean.map((m, i) => m + (std[i] || 0.001) * (1 + noise) * (Math.random() * 2 - 1));
}

// ============================================================
// MAIN COMPONENT
// ============================================================
export default function BiometricAuthDemo() {
  // --- State ---
  const [phase, setPhase] = useState("enroll"); // enroll | auth | attack
  const [enrollSamples, setEnrollSamples] = useState([]);
  const [userProfile, setUserProfile] = useState(null);
  const [text, setText] = useState("");
  const [keystrokeEvents, setKeystrokeEvents] = useState([]);
  const [mouseTrail, setMouseTrail] = useState([]);
  const [authResult, setAuthResult] = useState(null);
  const [attackResults, setAttackResults] = useState([]);
  const [attackType, setAttackType] = useState("fgsm");
  const [epsilon, setEpsilon] = useState(0.2);
  const [pgdSteps, setPgdSteps] = useState(20);
  const [mimicryNoise, setMimicryNoise] = useState(0.1);
  const [defenses, setDefenses] = useState({
    smoothing: false, squeezing: false, strictThreshold: false,
  });
  const [history, setHistory] = useState([]);
  const [mouseScore, setMouseScore] = useState(null);
  const [mouseProfile, setMouseProfile] = useState(null);
  const [mouseEnrollSamples, setMouseEnrollSamples] = useState([]);

  const inputRef = useRef(null);
  const mouseAreaRef = useRef(null);
  const currentKeys = useRef({});

  // --- Keystroke capture ---
  const handleKeyDown = useCallback((e) => {
    if (!currentKeys.current[e.key]) {
      currentKeys.current[e.key] = { key: e.key, down: performance.now() };
    }
  }, []);

  const handleKeyUp = useCallback((e) => {
    if (currentKeys.current[e.key]) {
      const event = { ...currentKeys.current[e.key], up: performance.now() };
      setKeystrokeEvents((prev) => [...prev, event]);
      delete currentKeys.current[e.key];
    }
  }, []);

  // --- Mouse capture ---
  const handleMouseMove = useCallback((e) => {
    const rect = mouseAreaRef.current?.getBoundingClientRect();
    if (!rect) return;
    setMouseTrail((prev) => {
      const next = [...prev, { x: e.clientX - rect.left, y: e.clientY - rect.top, t: performance.now() }];
      return next.length > 300 ? next.slice(-300) : next;
    });
  }, []);

  // --- Enroll ---
  const handleEnrollSample = () => {
    const features = extractKeystrokeFeatures(keystrokeEvents);
    if (!features) return;
    const newSamples = [...enrollSamples, features];
    setEnrollSamples(newSamples);

    // Also enroll mouse
    const mf = extractMouseFeatures(mouseTrail);
    if (mf) setMouseEnrollSamples((p) => [...p, mf]);

    setText("");
    setKeystrokeEvents([]);
    setMouseTrail([]);

    if (newSamples.length >= 5) {
      const mean = newSamples[0].map((_, i) =>
        newSamples.reduce((s, row) => s + row[i], 0) / newSamples.length
      );
      const std = newSamples[0].map((_, i) => {
        const m = mean[i];
        return Math.sqrt(
          newSamples.reduce((s, row) => s + (row[i] - m) ** 2, 0) / newSamples.length
        ) || 0.001;
      });
      setUserProfile({ mean, std, nSamples: newSamples.length });

      // Mouse profile
      if (mouseEnrollSamples.length >= 3) {
        const mm = mouseEnrollSamples[0].map((_, i) =>
          mouseEnrollSamples.reduce((s, r) => s + r[i], 0) / mouseEnrollSamples.length
        );
        const ms = mouseEnrollSamples[0].map((_, i) => {
          const m2 = mm[i];
          return Math.sqrt(
            mouseEnrollSamples.reduce((s, r) => s + (r[i] - m2) ** 2, 0) / mouseEnrollSamples.length
          ) || 0.001;
        });
        setMouseProfile({ mean: mm, std: ms });
      }
    }
  };

  const finishEnrollment = () => {
    if (userProfile) setPhase("auth");
  };

  // --- Authenticate ---
  const handleAuthenticate = () => {
    let features = extractKeystrokeFeatures(keystrokeEvents);
    if (!features || !userProfile) return;

    // Apply defenses
    if (defenses.smoothing) {
      features = features.map((v) => (v + v + (Math.random() - 0.5) * 0.1) / 2);
    }
    if (defenses.squeezing) {
      features = features.map((v) => Math.round(v * 16) / 16);
    }

    const score = gaussianScore(features, userProfile.mean, userProfile.std);
    const cosine = cosineSimilarity(features, userProfile.mean);
    const threshold = defenses.strictThreshold ? -2.0 : -5.0;
    const isGenuine = score > threshold && cosine > 0.5;

    // Mouse score
    let mScore = null;
    const mf = extractMouseFeatures(mouseTrail);
    if (mf && mouseProfile) {
      mScore = gaussianScore(mf, mouseProfile.mean, mouseProfile.std);
      setMouseScore(mScore);
    }

    const result = {
      score: Math.round(score * 1000) / 1000,
      cosine: Math.round(cosine * 1000) / 1000,
      mouseScore: mScore ? Math.round(mScore * 1000) / 1000 : null,
      isGenuine,
      threshold,
      timestamp: new Date().toLocaleTimeString(),
    };
    setAuthResult(result);
    setHistory((p) => [result, ...p].slice(0, 20));
    setText("");
    setKeystrokeEvents([]);
    setMouseTrail([]);
  };

  // --- Attack ---
  const runAttack = () => {
    if (!userProfile) return;
    const { mean, std } = userProfile;

    // Generate impostor baseline
    const impostor = mean.map((m, i) => m + (std[i] || 0.001) * 3 * (Math.random() * 2 - 1));
    const scoreBefore = gaussianScore(impostor, mean, std);

    let result;
    if (attackType === "fgsm") {
      const adv = fgsmAttack(impostor, mean, std, epsilon);
      const scoreAfter = gaussianScore(adv, mean, std);
      result = {
        type: "FGSM", epsilon,
        scoreBefore: Math.round(scoreBefore * 1000) / 1000,
        scoreAfter: Math.round(scoreAfter * 1000) / 1000,
        success: scoreAfter > -5.0,
        perturbation: Math.round(adv.reduce((s, v, i) => s + Math.abs(v - impostor[i]), 0) * 1000) / 1000,
      };
    } else if (attackType === "pgd") {
      const { adversarial, trajectory } = pgdAttack(impostor, mean, std, epsilon, pgdSteps);
      const scoreAfter = gaussianScore(adversarial, mean, std);
      result = {
        type: "PGD", epsilon, steps: pgdSteps,
        scoreBefore: Math.round(scoreBefore * 1000) / 1000,
        scoreAfter: Math.round(scoreAfter * 1000) / 1000,
        success: scoreAfter > -5.0,
        trajectory: trajectory.map((t) => Math.round(t * 100) / 100),
      };
    } else {
      const mim = mimicryAttack(mean, std, mimicryNoise);
      const scoreAfter = gaussianScore(mim, mean, std);
      result = {
        type: "Mimicry", noise: mimicryNoise,
        scoreBefore: 0,
        scoreAfter: Math.round(scoreAfter * 1000) / 1000,
        success: scoreAfter > -5.0,
      };
    }
    setAttackResults((p) => [result, ...p].slice(0, 15));
  };

  // ============================================================
  // RENDER
  // ============================================================
  const tabStyle = (t) => ({
    padding: "10px 20px", cursor: "pointer", fontWeight: 500, fontSize: 14,
    borderBottom: phase === t ? "2px solid #1D9E75" : "2px solid transparent",
    color: phase === t ? "#1D9E75" : "#888",
    background: "none", border: "none", borderBottom: phase === t ? "2px solid #1D9E75" : "2px solid transparent",
  });

  const cardStyle = {
    background: "var(--color-background-secondary, #f5f5f0)",
    borderRadius: 12, padding: "16px 20px", marginBottom: 12,
  };

  const btnPrimary = {
    padding: "10px 20px", borderRadius: 8, border: "none", fontWeight: 500,
    background: "#1D9E75", color: "#fff", cursor: "pointer", fontSize: 14,
  };

  const btnOutline = {
    padding: "8px 16px", borderRadius: 8, border: "1px solid #ccc",
    background: "transparent", cursor: "pointer", fontSize: 13,
  };

  const scoreBadge = (score, threshold = -5) => {
    const good = score > threshold;
    return (
      <span style={{
        display: "inline-block", padding: "3px 10px", borderRadius: 6, fontSize: 12, fontWeight: 500,
        background: good ? "#E1F5EE" : "#FCEBEB",
        color: good ? "#0F6E56" : "#A32D2D",
      }}>
        {good ? "Genuine" : "Impostor"} ({score})
      </span>
    );
  };

  return (
    <div style={{ fontFamily: "'DM Sans', 'Segoe UI', sans-serif", maxWidth: 720, margin: "0 auto", padding: 16 }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, margin: 0, color: "var(--color-text-primary, #1a1a1a)" }}>
          Biometric Authentication Demo
        </h1>
        <p style={{ fontSize: 13, color: "#888", margin: "4px 0 0" }}>
          Continuous authentication via keystroke dynamics + mouse behavior + adversarial attack simulation
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 0, borderBottom: "1px solid #e0e0e0", marginBottom: 20 }}>
        <button style={tabStyle("enroll")} onClick={() => setPhase("enroll")}>1. Enroll</button>
        <button style={tabStyle("auth")} onClick={() => userProfile && setPhase("auth")}>2. Authenticate</button>
        <button style={tabStyle("attack")} onClick={() => userProfile && setPhase("attack")}>3. Attack</button>
      </div>

      {/* ============ ENROLL ============ */}
      {phase === "enroll" && (
        <div>
          <div style={cardStyle}>
            <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 8 }}>
              Type the phrase below {enrollSamples.length}/5 times to create your biometric profile
            </div>
            <div style={{
              padding: "12px 16px", background: "var(--color-background-primary, #fff)",
              borderRadius: 8, fontSize: 15, fontWeight: 500, fontFamily: "monospace",
              marginBottom: 12, letterSpacing: 1, border: "1px solid #e0e0e0",
            }}>
              the quick brown fox
            </div>
            <input
              ref={inputRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              onKeyUp={handleKeyUp}
              placeholder="Start typing here..."
              style={{
                width: "100%", padding: "12px 16px", borderRadius: 8, fontSize: 15,
                border: "1px solid #ccc", outline: "none", boxSizing: "border-box",
              }}
            />
            <div
              ref={mouseAreaRef}
              onMouseMove={handleMouseMove}
              style={{
                height: 80, marginTop: 12, borderRadius: 8, border: "1px dashed #ccc",
                display: "flex", alignItems: "center", justifyContent: "center",
                color: "#aaa", fontSize: 12, position: "relative", overflow: "hidden",
              }}
            >
              Move your mouse here to capture movement pattern
              <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
                {mouseTrail.length > 1 && (
                  <polyline
                    points={mouseTrail.map((p) => `${p.x},${p.y}`).join(" ")}
                    fill="none" stroke="#1D9E75" strokeWidth="1.5" opacity="0.5"
                  />
                )}
              </svg>
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 12, alignItems: "center" }}>
              <button style={btnPrimary} onClick={handleEnrollSample}
                disabled={keystrokeEvents.length < 5}>
                Submit sample ({enrollSamples.length}/5)
              </button>
              {userProfile && (
                <button style={{ ...btnPrimary, background: "#085041" }} onClick={finishEnrollment}>
                  Finish enrollment
                </button>
              )}
            </div>
          </div>

          {/* Enrollment progress */}
          <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
            {[0, 1, 2, 3, 4].map((i) => (
              <div key={i} style={{
                flex: 1, height: 6, borderRadius: 3,
                background: i < enrollSamples.length ? "#1D9E75" : "#e0e0e0",
                transition: "background 0.3s",
              }} />
            ))}
          </div>
          {userProfile && (
            <div style={{ ...cardStyle, marginTop: 12, fontSize: 13, color: "#555" }}>
              Profile created from {userProfile.nSamples} samples with {userProfile.mean.length} features.
              {mouseProfile && " Mouse profile also captured."}
            </div>
          )}
        </div>
      )}

      {/* ============ AUTHENTICATE ============ */}
      {phase === "auth" && (
        <div>
          {/* Defense toggles */}
          <div style={{ ...cardStyle, display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: 13, fontWeight: 500 }}>Defenses:</span>
            {[
              { key: "smoothing", label: "Input smoothing" },
              { key: "squeezing", label: "Feature squeezing" },
              { key: "strictThreshold", label: "Strict threshold" },
            ].map(({ key, label }) => (
              <label key={key} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, cursor: "pointer" }}>
                <input type="checkbox" checked={defenses[key]}
                  onChange={() => setDefenses((d) => ({ ...d, [key]: !d[key] }))} />
                {label}
              </label>
            ))}
          </div>

          <div style={cardStyle}>
            <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 8 }}>
              Type the same phrase to authenticate
            </div>
            <input
              ref={inputRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              onKeyUp={handleKeyUp}
              placeholder="the quick brown fox"
              style={{
                width: "100%", padding: "12px 16px", borderRadius: 8, fontSize: 15,
                border: "1px solid #ccc", outline: "none", boxSizing: "border-box",
              }}
            />
            <div ref={mouseAreaRef} onMouseMove={handleMouseMove}
              style={{
                height: 60, marginTop: 10, borderRadius: 8, border: "1px dashed #ccc",
                position: "relative", overflow: "hidden",
              }}>
              <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
                {mouseTrail.length > 1 && (
                  <polyline points={mouseTrail.map((p) => `${p.x},${p.y}`).join(" ")}
                    fill="none" stroke="#1D9E75" strokeWidth="1.5" opacity="0.5" />
                )}
              </svg>
            </div>
            <button style={{ ...btnPrimary, marginTop: 12 }} onClick={handleAuthenticate}
              disabled={keystrokeEvents.length < 5}>
              Authenticate
            </button>
          </div>

          {/* Auth result */}
          {authResult && (
            <div style={{
              ...cardStyle,
              borderLeft: `4px solid ${authResult.isGenuine ? "#1D9E75" : "#E24B4A"}`,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: 16, fontWeight: 600, color: authResult.isGenuine ? "#0F6E56" : "#A32D2D" }}>
                  {authResult.isGenuine ? "Access granted" : "Access denied"}
                </span>
                <span style={{ fontSize: 12, color: "#999" }}>{authResult.timestamp}</span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginTop: 12 }}>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 20, fontWeight: 600 }}>{authResult.score}</div>
                  <div style={{ fontSize: 11, color: "#888" }}>Keystroke score</div>
                </div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 20, fontWeight: 600 }}>{authResult.cosine}</div>
                  <div style={{ fontSize: 11, color: "#888" }}>Cosine similarity</div>
                </div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 20, fontWeight: 600 }}>{authResult.mouseScore ?? "—"}</div>
                  <div style={{ fontSize: 11, color: "#888" }}>Mouse score</div>
                </div>
              </div>
            </div>
          )}

          {/* History */}
          {history.length > 0 && (
            <div style={{ ...cardStyle, maxHeight: 200, overflow: "auto" }}>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>Authentication history</div>
              {history.map((h, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", fontSize: 12, borderBottom: "1px solid #eee" }}>
                  <span>{h.timestamp}</span>
                  {scoreBadge(h.score, h.threshold)}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ============ ATTACK ============ */}
      {phase === "attack" && (
        <div>
          <div style={cardStyle}>
            <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 12 }}>
              Adversarial attack simulation
            </div>

            {/* Attack type selector */}
            <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
              {[
                { key: "fgsm", label: "FGSM", desc: "Single-step gradient" },
                { key: "pgd", label: "PGD", desc: "Iterative gradient" },
                { key: "mimicry", label: "Mimicry", desc: "Statistical copy" },
              ].map(({ key, label, desc }) => (
                <button key={key} onClick={() => setAttackType(key)}
                  style={{
                    ...btnOutline, flex: 1, textAlign: "center",
                    borderColor: attackType === key ? "#1D9E75" : "#ccc",
                    background: attackType === key ? "#E1F5EE" : "transparent",
                    color: attackType === key ? "#0F6E56" : "#555",
                  }}>
                  <div style={{ fontWeight: 500 }}>{label}</div>
                  <div style={{ fontSize: 11, marginTop: 2 }}>{desc}</div>
                </button>
              ))}
            </div>

            {/* Attack parameters */}
            {(attackType === "fgsm" || attackType === "pgd") && (
              <div style={{ marginBottom: 12 }}>
                <label style={{ fontSize: 13, color: "#666" }}>
                  Perturbation budget (ε): {epsilon}
                </label>
                <input type="range" min="0.01" max="2" step="0.01" value={epsilon}
                  onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                  style={{ width: "100%", marginTop: 4 }} />
              </div>
            )}
            {attackType === "pgd" && (
              <div style={{ marginBottom: 12 }}>
                <label style={{ fontSize: 13, color: "#666" }}>
                  PGD iterations: {pgdSteps}
                </label>
                <input type="range" min="5" max="100" step="5" value={pgdSteps}
                  onChange={(e) => setPgdSteps(parseInt(e.target.value))}
                  style={{ width: "100%", marginTop: 4 }} />
              </div>
            )}
            {attackType === "mimicry" && (
              <div style={{ marginBottom: 12 }}>
                <label style={{ fontSize: 13, color: "#666" }}>
                  Noise level: {mimicryNoise} (0 = perfect copy)
                </label>
                <input type="range" min="0" max="2" step="0.05" value={mimicryNoise}
                  onChange={(e) => setMimicryNoise(parseFloat(e.target.value))}
                  style={{ width: "100%", marginTop: 4 }} />
              </div>
            )}

            {/* Defense toggles for attack testing */}
            <div style={{ display: "flex", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
              {[
                { key: "smoothing", label: "Input smoothing" },
                { key: "squeezing", label: "Feature squeezing" },
                { key: "strictThreshold", label: "Strict threshold" },
              ].map(({ key, label }) => (
                <label key={key} style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                  <input type="checkbox" checked={defenses[key]}
                    onChange={() => setDefenses((d) => ({ ...d, [key]: !d[key] }))} />
                  {label}
                </label>
              ))}
            </div>

            <button style={btnPrimary} onClick={runAttack}>
              Launch {attackType.toUpperCase()} attack
            </button>
          </div>

          {/* Attack results */}
          {attackResults.length > 0 && (
            <div style={cardStyle}>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>Attack results</div>
              {attackResults.map((r, i) => (
                <div key={i} style={{
                  padding: "10px 12px", marginBottom: 6, borderRadius: 8,
                  background: "var(--color-background-primary, #fff)",
                  border: `1px solid ${r.success ? "#E24B4A" : "#5DCAA5"}`,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontWeight: 500, fontSize: 13 }}>{r.type}</span>
                    <span style={{
                      fontSize: 11, padding: "2px 8px", borderRadius: 4, fontWeight: 500,
                      background: r.success ? "#FCEBEB" : "#E1F5EE",
                      color: r.success ? "#A32D2D" : "#0F6E56",
                    }}>
                      {r.success ? "MODEL FOOLED" : "BLOCKED"}
                    </span>
                  </div>
                  <div style={{ display: "flex", gap: 16, marginTop: 6, fontSize: 12, color: "#666" }}>
                    <span>Before: {r.scoreBefore}</span>
                    <span>After: {r.scoreAfter}</span>
                    {r.epsilon && <span>ε={r.epsilon}</span>}
                    {r.noise !== undefined && <span>noise={r.noise}</span>}
                    {r.perturbation && <span>L1={r.perturbation}</span>}
                  </div>
                  {/* PGD trajectory mini-chart */}
                  {r.trajectory && (
                    <svg viewBox={`0 0 ${r.trajectory.length * 4} 40`} style={{ width: "100%", height: 30, marginTop: 4 }}>
                      <polyline
                        points={r.trajectory.map((v, j) => `${j * 4},${20 - Math.min(Math.max(v + 10, -20), 20)}`).join(" ")}
                        fill="none" stroke="#E24B4A" strokeWidth="1"
                      />
                      <line x1="0" y1="20" x2={r.trajectory.length * 4} y2="20"
                        stroke="#ccc" strokeWidth="0.5" strokeDasharray="2" />
                    </svg>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}