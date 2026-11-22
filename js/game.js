/* ==========================================================
   game.js
   Lógica de la partida (game.html): cámara, subida de foto,
   comunicación con FastAPI (/api/analysis/pose), mapa de calor,
   esqueleto coloreado por zona y ranking por rondas.
   ========================================================== */

(() => {
    "use strict";

    const $ = (id) => document.getElementById(id);
    const sculptures = window.MIMESIS_SCULPTURES || [];
    const ANALYSIS_ENDPOINT = "/api/analysis/pose";

    const fallbackState = {
        players: [
            { name: "Jugador 1", totalScore: 0, lastScore: null, roundsPlayed: 0 },
            { name: "Jugador 2", totalScore: 0, lastScore: null, roundsPlayed: 0 }
        ],
        sculptureId: sculptures[0]?.id || "belalcazar",
        sculpture: sculptures[0],
        currentPlayer: 0,
        round: 1,
        roundScores: []
    };

    let game;
    try {
        game = JSON.parse(sessionStorage.getItem("mimesisGame")) || fallbackState;
    } catch {
        game = fallbackState;
    }

    if (!Array.isArray(game.players) || game.players.length < 1) {
        game = fallbackState;
    }
    game.players = game.players.map((p, idx) => ({
        name: p.name || `Jugador ${idx + 1}`,
        totalScore: Number(p.totalScore) || 0,
        lastScore: typeof p.lastScore === "number" ? p.lastScore : null,
        roundsPlayed: Number(p.roundsPlayed) || 0
    }));
    game.sculpture =
        sculptures.find((s) => s.id === (game.sculptureId || game.sculpture?.id)) ||
        sculptures[0];
    game.sculptureId = game.sculpture.id;
    game.currentPlayer = Number.isInteger(game.currentPlayer) ? game.currentPlayer : 0;
    game.round = Number.isInteger(game.round) ? game.round : 1;
    game.roundScores = Array.isArray(game.roundScores) ? game.roundScores : [];

    let stream = null;
    let captured = false;
    let capturedBlob = null;
    let lastAnalysis = null;
    let countdownTimer = null;
    let toastTimer = null;

    /* Colores y definiciones de los 15 landmarks usados para calificar (coincide con extract_references.py) */
    const SEGMENT_COLORS = {
        cabeza: "#ffd71e",     // Amarillo dorado
        torso: "#ebebeb",      // Blanco / plata
        brazo_izq: "#32d7ff",  // Cian
        brazo_der: "#ff6e41",  // Naranja / coral
        pierna_izq: "#73eb5f", // Verde brillante
        pierna_der: "#dc64eb"  // Magenta / violeta
    };

    const SEGMENT_LEGEND_NAMES = {
        cabeza: "Cabeza",
        torso: "Torso",
        brazo_izq: "Brazo Izq",
        brazo_der: "Brazo Der",
        pierna_izq: "Pierna Izq",
        pierna_der: "Pierna Der"
    };

    const SEGMENT_CONNECTIONS = [
        [0, 7, "cabeza"], [0, 8, "cabeza"], [7, 11, "cabeza"], [8, 12, "cabeza"],
        [11, 12, "torso"], [11, 23, "torso"], [12, 24, "torso"], [23, 24, "torso"],
        [11, 13, "brazo_izq"], [13, 15, "brazo_izq"],
        [12, 14, "brazo_der"], [14, 16, "brazo_der"],
        [23, 25, "pierna_izq"], [25, 27, "pierna_izq"],
        [24, 26, "pierna_der"], [26, 28, "pierna_der"]
    ];

    const SCORING_LANDMARKS = [
        [0, "0:Nariz", "cabeza"],
        [7, "7:Oreja I", "cabeza"],
        [8, "8:Oreja D", "cabeza"],
        [11, "11:Hombro I", "brazo_izq"],
        [12, "12:Hombro D", "brazo_der"],
        [13, "13:Codo I", "brazo_izq"],
        [14, "14:Codo D", "brazo_der"],
        [15, "15:Muñeca I", "brazo_izq"],
        [16, "16:Muñeca D", "brazo_der"],
        [23, "23:Cadera I", "pierna_izq"],
        [24, "24:Cadera D", "pierna_der"],
        [25, "25:Rodilla I", "pierna_izq"],
        [26, "26:Rodilla D", "pierna_der"],
        [27, "27:Tobillo I", "pierna_izq"],
        [28, "28:Tobillo D", "pierna_der"]
    ];

    function scoreTier(value) {
        if (typeof value !== "number" || !Number.isFinite(value)) return "na";
        if (value >= 80) return "high";
        if (value >= 55) return "mid";
        return "low";
    }

    function tierColor(tier) {
        const map = {
            high: "#6ea884",
            mid: "#d6a44e",
            low: "#c96857",
            na: "#8f867b"
        };
        return map[tier] || map.na;
    }

    function notify(message) {
        const el = $("toast");
        el.textContent = message;
        el.classList.add("show");
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => el.classList.remove("show"), 3200);
    }

    function currentPlayer() {
        return game.players[game.currentPlayer];
    }

    function saveState() {
        sessionStorage.setItem("mimesisGame", JSON.stringify(game));
    }

    async function checkBackend() {
        const dot = $("backend-dot");
        const label = $("backend-label");
        try {
            const res = await fetch("/api/health", { cache: "no-store" });
            if (!res.ok) throw new Error();
            const data = await res.json();
            dot.className = "status-dot online";
            label.textContent = data.mlp_loaded
                ? "Python · MLP activo"
                : "Python · Modo geométrico";
        } catch {
            dot.className = "status-dot offline";
            label.textContent = "Python desconectado";
        }
    }

    function renderGame() {
        const player = currentPlayer();
        const s = game.sculpture;

        $("current-player-name").textContent = player.name;
        $("current-avatar").textContent = player.name.trim().charAt(0).toUpperCase();
        $("player-position").textContent =
            `${String(game.currentPlayer + 1).padStart(2, "0")} / ${String(game.players.length).padStart(2, "0")}`;
        $("round-number").textContent = String(game.round).padStart(2, "0");
        $("turn-counter").textContent = `TURNO ${game.currentPlayer + 1}`;

        $("sculpture-title").textContent = s.title;
        $("sculpture-author").textContent = s.author;
        $("sculpture-emotion").textContent = s.emotion;
        window.setSculptureImage($("sculpture-image"), s);
        const refStage = $("reference-result-stage");
        if (refStage) {
            refStage.innerHTML = '<img id="reference-result-image" alt="Obra de referencia">';
            window.setSculptureImage($("reference-result-image"), s);
        }
        $("cultural-description").textContent = s.description;

        const list = $("player-list");
        list.innerHTML = "";
        game.players.forEach((p, idx) => {
            const row = document.createElement("div");
            row.className = `turn-row${idx === game.currentPlayer ? " active" : ""}`;

            const av = document.createElement("span");
            av.className = "avatar";
            av.textContent = p.name.trim().charAt(0).toUpperCase();

            const name = document.createElement("span");
            name.className = "turn-name";
            name.textContent = p.name;

            const score = document.createElement("span");
            score.className = "turn-score";
            score.textContent = p.roundsPlayed > 0 ? `${Math.round(p.totalScore)} pts` : "—";

            row.append(av, name, score);
            list.appendChild(row);
        });

        const progress = Math.round((game.currentPlayer / game.players.length) * 100);
        $("progress-text").textContent = `${progress}%`;
        $("progress-fill").style.width = `${progress}%`;
    }

    /* ================= CÁMARA Y SUBIDA DE FOTO ================= */

    let selectedDeviceId = "";
    let cameraRotation = 0; // 0, 90, 180, 270 grados
    let mirrorCamera = true; // Efecto espejo por defecto (ideal para selfie)

    function updateVideoTransform() {
        const video = $("camera-video");
        const frame = video.closest(".camera-frame");
        const vw = video.videoWidth || 0;
        const vh = video.videoHeight || 0;
        const swapped = cameraRotation === 90 || cameraRotation === 270;
        const effectiveW = swapped ? vh : vw;
        const effectiveH = swapped ? vw : vh;
        const isPortrait = effectiveH > effectiveW && effectiveH > 0;

        if (frame) {
            frame.classList.toggle("portrait", isPortrait);
        }

        const parts = [];
        if (mirrorCamera) parts.push("scaleX(-1)");
        if (cameraRotation !== 0) parts.push(`rotate(${cameraRotation}deg)`);
        video.style.transform = parts.join(" ");

        $("rotate-btn").textContent = `Rotar ${cameraRotation}°`;
        $("mirror-btn").textContent = `Espejo: ${mirrorCamera ? "Sí" : "No"}`;
        $("mirror-btn").classList.toggle("active", mirrorCamera);
    }

    async function refreshCameraList(activeDeviceId = null) {
        const select = $("camera-select");
        if (!select || !navigator.mediaDevices?.enumerateDevices) return;
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const cameras = devices.filter((d) => d.kind === "videoinput");
            const prevValue = activeDeviceId || selectedDeviceId || select.value;

            select.innerHTML = "";
            if (cameras.length === 0) {
                const opt = document.createElement("option");
                opt.value = "";
                opt.textContent = "Sin cámaras detectadas";
                select.appendChild(opt);
                selectedDeviceId = "";
                return;
            }

            cameras.forEach((cam, idx) => {
                const opt = document.createElement("option");
                opt.value = cam.deviceId;
                opt.textContent = cam.label || `Cámara ${idx + 1}`;
                select.appendChild(opt);
            });

            if (prevValue && cameras.some((c) => c.deviceId === prevValue)) {
                select.value = prevValue;
                selectedDeviceId = prevValue;
            } else {
                selectedDeviceId = cameras[0].deviceId;
                select.value = selectedDeviceId;
            }
        } catch {
            // Ignorar si el navegador aún no permite enumerar dispositivos
        }
    }

    async function startCamera(explicitDeviceId = null) {
        if (!navigator.mediaDevices?.getUserMedia) {
            notify("El navegador requiere HTTPS o localhost para usar la cámara.");
            return;
        }

        // Cerrar stream anterior si estamos cambiando de cámara en vivo
        if (stream) {
            stream.getTracks().forEach((t) => t.stop());
            stream = null;
        }

        const targetId = explicitDeviceId ?? ($("camera-select")?.value || selectedDeviceId);
        // No forzar resolución cuando se elige un dispositivo concreto (evita pantalla verde en cámaras virtuales como DroidCam)
        const videoConstraints = targetId
            ? { deviceId: { exact: targetId } }
            : { facingMode: "user" };

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: videoConstraints,
                audio: false
            });
            const track = stream.getVideoTracks()[0];
            const activeId = track?.getSettings?.().deviceId || targetId || "";
            selectedDeviceId = activeId;

            const video = $("camera-video");
            video.srcObject = stream;
            video.hidden = false;
            video.onloadedmetadata = () => updateVideoTransform();
            updateVideoTransform();

            $("camera-placeholder").hidden = true;
            $("camera-hud").hidden = false;
            $("captured-image").hidden = true;
            $("camera-status").textContent = track?.label ? `Activa · ${track.label}` : "Cámara activa";
            $("camera-btn").textContent = "Desactivar cámara";
            $("capture-btn").disabled = false;
            captured = false;
            $("retake-btn").hidden = true;
            $("result-panel").classList.remove("visible");

            await refreshCameraList(activeId);
        } catch (err) {
            // Si falló con un deviceId guardado (ej. celular desconectado), reintentar con la predeterminada
            if (targetId && (err.name === "OverconstrainedError" || err.name === "NotFoundError")) {
                selectedDeviceId = "";
                return startCamera("");
            }
            notify(
                err.name === "NotAllowedError"
                    ? "Permiso de cámara denegado."
                    : "No se pudo conectar con la cámara seleccionada."
            );
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach((t) => t.stop());
            stream = null;
        }
        $("camera-video").srcObject = null;
        $("camera-video").hidden = true;
        if (!captured) $("camera-placeholder").hidden = false;
        $("camera-hud").hidden = true;
        $("camera-status").textContent = "Cámara desconectada";
        $("camera-btn").textContent = "Activar cámara";
        $("capture-btn").disabled = true;
    }

    function beginCapture() {
        if (!stream || captured) return;
        $("capture-btn").disabled = true;
        $("camera-btn").disabled = true;
        let count = 3;
        $("countdown").textContent = count;
        $("countdown").classList.add("visible");

        countdownTimer = setInterval(() => {
            count -= 1;
            if (count > 0) {
                $("countdown").textContent = count;
            } else {
                clearInterval(countdownTimer);
                $("countdown").classList.remove("visible");
                takeSnapshot();
            }
        }, 1000);
    }

    function showCapturedImage(dataUrl, blob) {
        capturedBlob = blob;
        captured = true;
        lastAnalysis = null;
        $("captured-image").src = dataUrl;
        $("captured-image").hidden = false;
        $("camera-video").hidden = true;
        $("camera-placeholder").hidden = true;
        $("camera-hud").hidden = true;
        $("camera-status").textContent = "Fotografía lista";
        $("camera-btn").disabled = false;
        $("capture-btn").disabled = true;
        $("retake-btn").hidden = false;
        $("result-panel").classList.add("visible");
        resetAnalysisView();
        $("result-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function takeSnapshot() {
        const video = $("camera-video");
        if (!video.videoWidth || !video.videoHeight) {
            notify("La cámara aún está iniciando. Intenta de nuevo.");
            $("capture-btn").disabled = false;
            $("camera-btn").disabled = false;
            return;
        }

        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const swapped = cameraRotation === 90 || cameraRotation === 270;
        const canvas = $("capture-canvas");
        canvas.width = swapped ? vh : vw;
        canvas.height = swapped ? vw : vh;

        const ctx = canvas.getContext("2d");
        ctx.save();
        ctx.translate(canvas.width / 2, canvas.height / 2);
        if (mirrorCamera) ctx.scale(-1, 1);
        if (cameraRotation !== 0) ctx.rotate((cameraRotation * Math.PI) / 180);
        ctx.drawImage(video, -vw / 2, -vh / 2, vw, vh);
        ctx.restore();

        const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
        canvas.toBlob((blob) => showCapturedImage(dataUrl, blob), "image/jpeg", 0.92);
    }

    function handleUpload(event) {
        const file = event.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => showCapturedImage(reader.result, file);
        reader.readAsDataURL(file);
        event.target.value = "";
    }

    /* ================= VISUALIZACIÓN Y ANÁLISIS ================= */

    function resetAnalysisView() {
        $("analysis-status").textContent = "Pendiente";
        $("analysis-status").className = "status-badge";
        $("processing-box").classList.remove("visible");
        $("analyze-btn").disabled = false;
        $("analyze-btn").textContent = "Evaluar pose con Python";
        $("result-message").textContent =
            "Fotografía lista. Presiona «Evaluar pose con Python» para ejecutar el preprocesamiento PDI y BlazePose.";
        $("score-value").textContent = "—";
        $("score-value").className = "score-ring";
        $("score-title").textContent = "Sin evaluar";
        $("score-description").textContent =
            "El puntaje de 0 a 100 aparecerá aquí al finalizar el análisis.";
        $("player-result-stage").innerHTML =
            '<div class="visual-placeholder">El esqueleto coloreado por zona se dibujará aquí.</div>';
        renderHeatmap(null);
        renderPdiDiagnostics(null);
    }

    function boxesOverlap(a, b, pad = 4) {
        return !(
            a.x1 + pad < b.x0 ||
            b.x1 + pad < a.x0 ||
            a.y1 + pad < b.y0 ||
            b.y1 + pad < a.y0
        );
    }

    function drawSegmentLegend(ctx, scale) {
        const fontSize = Math.max(11, Math.round(13 * scale));
        const padX = Math.max(8, Math.round(10 * scale));
        const padY = Math.max(6, Math.round(8 * scale));
        const swatchR = Math.max(4, Math.round(5.5 * scale));
        const rowGap = Math.max(5, Math.round(7 * scale));
        const rowH = fontSize + rowGap;

        ctx.font = `600 ${fontSize}px system-ui, -apple-system, sans-serif`;
        const entries = Object.entries(SEGMENT_COLORS).map(([seg, color]) => ({
            name: SEGMENT_LEGEND_NAMES[seg],
            color
        }));

        const maxTw = Math.max(...entries.map((e) => ctx.measureText(e.name).width));
        const x0 = padX;
        const y0 = padY;
        const boxW = padX * 2 + swatchR * 2 + Math.round(8 * scale) + maxTw;
        const boxH = padY * 2 + rowH * entries.length - Math.round(rowGap / 2);

        ctx.fillStyle = "rgba(18, 17, 15, 0.78)";
        ctx.fillRect(x0, y0, boxW, boxH);
        ctx.strokeStyle = "rgba(140, 140, 140, 0.85)";
        ctx.lineWidth = Math.max(1, 1.2 * scale);
        ctx.strokeRect(x0, y0, boxW, boxH);

        ctx.textBaseline = "middle";
        entries.forEach((entry, i) => {
            const cy = y0 + padY + i * rowH + fontSize * 0.5;
            const cx = x0 + padX + swatchR;

            ctx.beginPath();
            ctx.arc(cx, cy, swatchR, 0, Math.PI * 2);
            ctx.fillStyle = entry.color;
            ctx.fill();
            ctx.lineWidth = Math.max(1, 1.2 * scale);
            ctx.strokeStyle = "#12110f";
            ctx.stroke();

            ctx.fillStyle = "#f5f5f5";
            ctx.fillText(entry.name, cx + swatchR + Math.round(8 * scale), cy);
        });

        return { x0, y0, x1: x0 + boxW, y1: y0 + boxH };
    }

    function drawPoseOverlay(container, imageUrl, landmarks) {
        container.innerHTML = "";
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            const canvas = document.createElement("canvas");
            canvas.width = img.naturalWidth || 800;
            canvas.height = img.naturalHeight || 600;
            canvas.style.width = "100%";
            canvas.style.height = "100%";
            canvas.style.objectFit = "contain";
            canvas.style.display = "block";

            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            if (Array.isArray(landmarks) && landmarks.length) {
                const w = canvas.width;
                const h = canvas.height;
                const scale = Math.max(0.65, Math.max(w, h) / 1200);
                const lineW = Math.max(2.5, 4.2 * scale);
                const outlineW = lineW + Math.max(2, 2.8 * scale);
                const dotR = Math.max(4.5, 6.5 * scale);

                // 1. Conexiones coloreadas por segmento (con borde oscuro)
                ctx.lineCap = "round";
                SEGMENT_CONNECTIONS.forEach(([a, b, segment]) => {
                    const pA = landmarks[a];
                    const pB = landmarks[b];
                    if (!pA || !pB) return;
                    if ((pA.visibility ?? 1) < 0.3 || (pB.visibility ?? 1) < 0.3) return;

                    const x1 = pA.x * w;
                    const y1 = pA.y * h;
                    const x2 = pB.x * w;
                    const y2 = pB.y * h;

                    ctx.strokeStyle = "#12110f";
                    ctx.lineWidth = outlineW;
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();

                    ctx.strokeStyle = SEGMENT_COLORS[segment];
                    ctx.lineWidth = lineW;
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                });

                // 2. Solo los 15 puntos clave usados para calificar
                const visiblePoints = [];
                SCORING_LANDMARKS.forEach(([idx, label, segment]) => {
                    const pt = landmarks[idx];
                    if (!pt || typeof pt.x !== "number" || typeof pt.y !== "number") return;
                    if ((pt.visibility ?? 1) < 0.3) return;

                    const px = pt.x * w;
                    const py = pt.y * h;
                    const color = SEGMENT_COLORS[segment];

                    ctx.beginPath();
                    ctx.arc(px, py, dotR, 0, Math.PI * 2);
                    ctx.fillStyle = color;
                    ctx.fill();
                    ctx.lineWidth = Math.max(1.5, 2 * scale);
                    ctx.strokeStyle = "#12110f";
                    ctx.stroke();

                    visiblePoints.push({ idx, label, segment, color, px, py });
                });

                // 3. Leyenda en esquina superior izquierda
                const legendBox = drawSegmentLegend(ctx, scale);

                // 4. Etiquetas legibles sin solapamiento
                const fontSize = Math.max(10, Math.round(12 * scale));
                const padX = Math.max(5, Math.round(6 * scale));
                const padY = Math.max(3, Math.round(4 * scale));
                const margin = Math.max(6, Math.round(8 * scale));
                const baseD = Math.max(14, Math.round(18 * scale));

                ctx.font = `600 ${fontSize}px system-ui, -apple-system, sans-serif`;
                ctx.textBaseline = "middle";

                const bodyCx = visiblePoints.reduce((acc, p) => acc + p.px, 0) / Math.max(1, visiblePoints.length);
                const placedBoxes = [legendBox];
                const pointBoxes = visiblePoints.map((p) => ({
                    x0: p.px - dotR - 2,
                    y0: p.py - dotR - 2,
                    x1: p.px + dotR + 2,
                    y1: p.py + dotR + 2
                }));

                const ordered = [...visiblePoints].sort((a, b) => {
                    const ga = a.segment === "cabeza" ? 0 : 1;
                    const gb = b.segment === "cabeza" ? 0 : 1;
                    return ga - gb || a.py - b.py || a.px - b.px;
                });

                ordered.forEach((p) => {
                    const tw = ctx.measureText(p.label).width;
                    const boxW = tw + padX * 2;
                    const boxH = fontSize + padY * 2;

                    const horizDirs = p.px >= bodyCx ? [1, -1] : [-1, 1];
                    const candidates = [];
                    [1.0, 2.2, 3.6, 5.2, 7.0].forEach((mult) => {
                        const d = baseD * mult;
                        horizDirs.forEach((hdir) => {
                            const bx = hdir > 0 ? p.px + d : p.px - d - boxW;
                            candidates.push([bx, p.py - boxH / 2]);
                            candidates.push([bx, p.py - boxH - d * 0.45]);
                            candidates.push([bx, p.py + d * 0.45]);
                        });
                        candidates.push([p.px - boxW / 2, p.py - d - boxH]);
                        candidates.push([p.px - boxW / 2, p.py + d]);
                    });

                    let chosen = null;
                    for (const [bx, by] of candidates) {
                        const x0 = Math.max(margin, Math.min(w - margin - boxW, bx));
                        const y0 = Math.max(margin, Math.min(h - margin - boxH, by));
                        const cand = { x0, y0, x1: x0 + boxW, y1: y0 + boxH };
                        if (placedBoxes.some((pb) => boxesOverlap(cand, pb, 4))) continue;
                        if (pointBoxes.some((ptb) => boxesOverlap(cand, ptb, 2))) continue;
                        chosen = cand;
                        break;
                    }

                    if (!chosen) {
                        for (const [bx, by] of candidates) {
                            const x0 = Math.max(margin, Math.min(w - margin - boxW, bx));
                            const y0 = Math.max(margin, Math.min(h - margin - boxH, by));
                            const cand = { x0, y0, x1: x0 + boxW, y1: y0 + boxH };
                            if (!placedBoxes.some((pb) => boxesOverlap(cand, pb, 2))) {
                                chosen = cand;
                                break;
                            }
                        }
                    }

                    if (!chosen) {
                        const x0 = Math.max(margin, Math.min(w - margin - boxW, p.px + baseD));
                        const y0 = Math.max(margin, Math.min(h - margin - boxH, p.py - boxH / 2));
                        chosen = { x0, y0, x1: x0 + boxW, y1: y0 + boxH };
                    }

                    placedBoxes.push(chosen);

                    const anchorX = Math.max(chosen.x0, Math.min(chosen.x1, p.px));
                    const anchorY = Math.max(chosen.y0, Math.min(chosen.y1, p.py));

                    ctx.strokeStyle = "#12110f";
                    ctx.lineWidth = Math.max(2, 2.5 * scale);
                    ctx.beginPath();
                    ctx.moveTo(p.px, p.py);
                    ctx.lineTo(anchorX, anchorY);
                    ctx.stroke();

                    ctx.strokeStyle = p.color;
                    ctx.lineWidth = Math.max(1, 1.3 * scale);
                    ctx.beginPath();
                    ctx.moveTo(p.px, p.py);
                    ctx.lineTo(anchorX, anchorY);
                    ctx.stroke();

                    ctx.fillStyle = "rgba(18, 17, 15, 0.82)";
                    ctx.fillRect(chosen.x0, chosen.y0, boxW, boxH);
                    ctx.strokeStyle = p.color;
                    ctx.lineWidth = Math.max(1, 1.5 * scale);
                    ctx.strokeRect(chosen.x0, chosen.y0, boxW, boxH);

                    ctx.fillStyle = "#f8f8f8";
                    ctx.fillText(p.label, chosen.x0 + padX, chosen.y0 + boxH / 2);
                });
            }

            container.innerHTML = "";
            container.appendChild(canvas);
        };
        img.src = imageUrl;
    }

    function renderHeatmap(regions) {
        const labels = { head: "Cabeza", torso: "Torso", arms: "Brazos", legs: "Piernas" };
        const host = $("heatmap-regions");
        host.innerHTML = "";

        Object.entries(labels).forEach(([key, label]) => {
            const raw = regions?.[key];
            const value = typeof raw === "number" ? raw : raw?.similarity ?? null;
            const tier = scoreTier(value);

            const row = document.createElement("div");
            row.className = "heat-row";

            const name = document.createElement("span");
            name.textContent = label;

            const track = document.createElement("div");
            track.className = "heat-track";

            const fill = document.createElement("div");
            fill.className = `heat-fill ${tier}`;
            fill.style.width = value === null ? "0%" : `${Math.max(0, Math.min(100, value))}%`;
            track.appendChild(fill);

            const num = document.createElement("strong");
            num.textContent = value === null ? "—" : `${Math.round(value)}%`;

            row.append(name, track, num);
            host.appendChild(row);
        });
    }

    function renderPdiDiagnostics(pdi) {
        $("pdi-sharpness").textContent = pdi
            ? `${Math.round(pdi.laplacian_var)} (${pdi.is_blurry ? "Borrosa" : "Nítida"})`
            : "—";
        $("pdi-clahe").textContent = pdi
            ? (pdi.clahe_applied ? `Sí (L=${Math.round(pdi.mean_luminance)})` : `No (L=${Math.round(pdi.mean_luminance)})`)
            : "—";
        $("pdi-confidence").textContent = pdi && typeof pdi.mean_visibility === "number"
            ? `${Math.round(pdi.mean_visibility * 100)}%`
            : "—";
    }

    function applyAnalysisResult(data) {
        const score = Number(data.score);
        const tier = scoreTier(score);

        if (Number.isFinite(score)) {
            $("score-value").textContent = `${Math.round(score)}`;
            $("score-value").className = `score-ring ${tier}`;
            $("score-title").textContent =
                tier === "high" ? "Imitación excelente"
                : tier === "mid" ? "Muy cerca de la obra"
                : "Ajusta tu postura";
            $("score-description").textContent = data.feedback || "Evaluación completada.";
        }

        const baseImage = data.annotated_image || $("captured-image").src;
        if (Array.isArray(data.landmarks) && data.landmarks.length) {
            drawPoseOverlay($("player-result-stage"), baseImage, data.landmarks);
        } else {
            $("player-result-stage").innerHTML = "";
            const img = document.createElement("img");
            img.src = baseImage;
            $("player-result-stage").appendChild(img);
        }

        if (Array.isArray(data.reference_landmarks) && data.reference_landmarks.length) {
            drawPoseOverlay(
                $("reference-result-stage"),
                game.sculpture.image,
                data.reference_landmarks
            );
        }

        renderHeatmap(data.regions || null);
        renderPdiDiagnostics(data.pdi_metrics || null);

        $("analysis-status").textContent = data.evaluator_mode === "mlp"
            ? "Evaluado (MLP propio)"
            : "Evaluado (geométrico)";
        $("analysis-status").className = "status-badge ready";
        $("result-message").textContent = data.message || "Análisis completado.";
        lastAnalysis = data;
    }

    async function processCapture() {
        if (!captured || !capturedBlob) {
            notify("Primero captura o sube una fotografía.");
            return;
        }

        const btn = $("analyze-btn");
        btn.disabled = true;
        $("processing-box").classList.add("visible");
        $("analysis-status").textContent = "Procesando";
        $("analysis-status").className = "status-badge";

        try {
            const form = new FormData();
            form.append("image", capturedBlob, "capture.jpg");
            form.append("player", currentPlayer().name);
            form.append("sculpture_id", game.sculpture.id);
            form.append("sculpture", JSON.stringify(game.sculpture));
            form.append("round", String(game.round));

            const response = await fetch(ANALYSIS_ENDPOINT, {
                method: "POST",
                body: form
            });

            const payload = await response.json().catch(() => null);
            if (!response.ok) {
                throw new Error(payload?.detail || `HTTP ${response.status}`);
            }

            applyAnalysisResult(payload);
        } catch (err) {
            $("analysis-status").textContent = "Error";
            $("analysis-status").className = "status-badge error";
            $("result-message").textContent =
                `No se pudo completar el análisis (${err.message}). Verifica que el servidor Python esté ejecutándose en localhost:8000.`;
            notify("Error al conectar con el backend Python.");
        } finally {
            $("processing-box").classList.remove("visible");
            btn.disabled = false;
            btn.textContent = lastAnalysis ? "Volver a evaluar" : "Reintentar evaluación";
        }
    }

    /* ================= TURNOS Y RANKING ================= */

    function retake() {
        captured = false;
        capturedBlob = null;
        lastAnalysis = null;
        resetAnalysisView();
        $("captured-image").hidden = true;
        $("result-panel").classList.remove("visible");
        $("retake-btn").hidden = true;
        if (stream) {
            $("camera-video").hidden = false;
            $("camera-hud").hidden = false;
            $("camera-status").textContent = "Cámara activa";
            $("capture-btn").disabled = false;
        } else {
            $("camera-placeholder").hidden = false;
            $("camera-status").textContent = "Cámara desconectada";
        }
    }

    function openRankingModal() {
        const tbody = $("ranking-body");
        tbody.innerHTML = "";
        const sorted = [...game.players].sort((a, b) => b.totalScore - a.totalScore);
        sorted.forEach((p, idx) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>#${idx + 1}</td>
                <td><strong>${p.name}</strong></td>
                <td class="num">${p.lastScore !== null ? Math.round(p.lastScore) : "—"}</td>
                <td class="num"><strong>${Math.round(p.totalScore)}</strong></td>
            `;
            tbody.appendChild(tr);
        });
        $("ranking-round-label").textContent = `Ronda ${game.round} completada · ${game.sculpture.title}`;
        $("ranking-modal").classList.add("open");
    }

    function confirmTurn() {
        const turnScore = typeof lastAnalysis?.score === "number" ? lastAnalysis.score : 0;
        const player = currentPlayer();
        player.lastScore = lastAnalysis ? turnScore : null;
        if (lastAnalysis) {
            player.totalScore += turnScore;
            player.roundsPlayed += 1;
        }

        const isLastPlayer = game.currentPlayer >= game.players.length - 1;
        if (isLastPlayer) {
            saveState();
            renderGame();
            openRankingModal();
            return;
        }

        game.currentPlayer += 1;
        saveState();
        retake();
        renderGame();
        notify(`Turno de ${currentPlayer().name}.`);
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function startNextRound() {
        $("ranking-modal").classList.remove("open");
        game.round += 1;
        game.currentPlayer = 0;
        // Rota automáticamente a la siguiente escultura de las 3 disponibles
        const nextIdx = (sculptures.findIndex((s) => s.id === game.sculpture.id) + 1) % sculptures.length;
        game.sculpture = sculptures[nextIdx];
        game.sculptureId = game.sculpture.id;
        saveState();
        retake();
        renderGame();
        notify(`Ronda ${game.round}: ahora imitan ${game.sculpture.title}.`);
    }

    /* Eventos */
    $("camera-btn").addEventListener("click", () => (stream ? stopCamera() : startCamera()));
    $("camera-select").addEventListener("change", (e) => {
        selectedDeviceId = e.target.value;
        if (stream) startCamera(selectedDeviceId);
    });
    $("refresh-cameras-btn").addEventListener("click", async () => {
        await refreshCameraList();
        notify("Lista de cámaras actualizada.");
    });
    $("rotate-btn").addEventListener("click", () => {
        cameraRotation = (cameraRotation + 90) % 360;
        updateVideoTransform();
    });
    $("mirror-btn").addEventListener("click", () => {
        mirrorCamera = !mirrorCamera;
        updateVideoTransform();
    });
    navigator.mediaDevices?.addEventListener?.("devicechange", () => refreshCameraList());

    $("capture-btn").addEventListener("click", beginCapture);
    $("upload-input").addEventListener("change", handleUpload);
    $("retake-btn").addEventListener("click", retake);
    $("analyze-btn").addEventListener("click", processCapture);
    $("next-player-btn").addEventListener("click", confirmTurn);
    $("next-round-btn").addEventListener("click", startNextRound);
    $("finish-game-btn").addEventListener("click", () => {
        stopCamera();
        window.location.href = "index.html";
    });
    $("back-button").addEventListener("click", () => {
        stopCamera();
        window.location.href = "index.html";
    });
    window.addEventListener("beforeunload", stopCamera);

    renderGame();
    checkBackend();
    refreshCameraList();
})();
