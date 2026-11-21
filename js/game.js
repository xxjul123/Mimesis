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

    /* Conexiones de los 33 landmarks de MediaPipe agrupadas por región corporal */
    const REGION_CONNECTIONS = {
        head: [[0, 7], [0, 8], [7, 11], [8, 12]],
        torso: [[11, 12], [11, 23], [12, 24], [23, 24]],
        arms: [
            [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
            [12, 14], [14, 16], [16, 18], [16, 20], [16, 22]
        ],
        legs: [
            [23, 25], [25, 27], [27, 29], [27, 31],
            [24, 26], [26, 28], [28, 30], [28, 32]
        ]
    };

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

    async function startCamera() {
        if (!navigator.mediaDevices?.getUserMedia) {
            notify("El navegador requiere HTTPS o localhost para usar la cámara.");
            return;
        }
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false
            });
            $("camera-video").srcObject = stream;
            $("camera-video").hidden = false;
            $("camera-placeholder").hidden = true;
            $("camera-hud").hidden = false;
            $("captured-image").hidden = true;
            $("camera-status").textContent = "Cámara activa";
            $("camera-btn").textContent = "Desactivar cámara";
            $("capture-btn").disabled = false;
            captured = false;
            $("retake-btn").hidden = true;
            $("result-panel").classList.remove("visible");
        } catch (err) {
            notify(
                err.name === "NotAllowedError"
                    ? "Permiso de cámara denegado."
                    : "No se pudo conectar con la cámara."
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
        const canvas = $("capture-canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

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

    function drawPoseOverlay(container, imageUrl, landmarks, regions, uniformColor = null) {
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
                const lineW = Math.max(2.5, Math.min(w, h) * 0.0055);
                const dotR = Math.max(3.5, Math.min(w, h) * 0.0075);

                Object.entries(REGION_CONNECTIONS).forEach(([regionKey, pairs]) => {
                    const raw = regions?.[regionKey];
                    const val = typeof raw === "number" ? raw : raw?.similarity;
                    ctx.strokeStyle = uniformColor || tierColor(scoreTier(val));
                    ctx.lineWidth = lineW;
                    ctx.lineCap = "round";

                    pairs.forEach(([a, b]) => {
                        const pA = landmarks[a];
                        const pB = landmarks[b];
                        if (!pA || !pB) return;
                        if ((pA.visibility ?? 1) < 0.3 || (pB.visibility ?? 1) < 0.3) return;
                        ctx.beginPath();
                        ctx.moveTo(pA.x * w, pA.y * h);
                        ctx.lineTo(pB.x * w, pB.y * h);
                        ctx.stroke();
                    });
                });

                landmarks.forEach((pt) => {
                    if (typeof pt.x !== "number" || typeof pt.y !== "number") return;
                    if ((pt.visibility ?? 1) < 0.3) return;
                    ctx.beginPath();
                    ctx.arc(pt.x * w, pt.y * h, dotR, 0, Math.PI * 2);
                    ctx.fillStyle = "#f3eee6";
                    ctx.fill();
                    ctx.lineWidth = Math.max(1.5, lineW * 0.45);
                    ctx.strokeStyle = "#12110f";
                    ctx.stroke();
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
            drawPoseOverlay($("player-result-stage"), baseImage, data.landmarks, data.regions);
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
                data.reference_landmarks,
                null,
                "#c88d55"
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
})();
