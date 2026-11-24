/* ==========================================================
   auto_mode.js
   Modo Automático en Pantalla Completa para MIMESIS:
   - Flujo sin botones por las 3 esculturas de la colección
   - 10s de preparación por jugador (nombre arriba, obra en esquina)
   - 5s de cambio de jugador evaluando en segundo plano
   - 8s de resultados comparativos con Pausa, Informe Detallado
     (que congela el reloj y lo reanuda al cerrar si no se pausó)
     y acceso directo al Inspector del Pipeline PDI ampliable
   - Podio final + avisos sonoros sutiles (Web Audio API)
   ========================================================== */

(() => {
    "use strict";

    const $ = (id) => document.getElementById(id);
    const bridge = window.MimesisGameBridge;
    if (!bridge) {
        $("open-auto-mode-btn")?.addEventListener("click", () => {
            window.location.reload();
        });
        return;
    }

    const POSE_SECONDS = 10;
    const SWAP_SECONDS = 5;
    const RESULTS_SECONDS = 8;
    const RING_CIRCUMFERENCE = 2 * Math.PI * 36; // r=36 en el SVG

    let isOpen = false;
    let isRunning = false;
    let activeRunToken = 0;
    let soundEnabled = true;
    let audioCtx = null;

    // Estado de la fase de resultados (8s)
    let resultsUserPaused = false;
    let resultsSkipRequested = false;
    let resultsRemainingMs = RESULTS_SECONDS * 1000;
    let resultsNextLabelBase = "";

    // Historial de rondas de la partida automática y estado del Informe Detallado
    let sessionHistory = [];
    let detailRoundRecord = null;
    let detailPlayerIndex = 0;
    let detailViewMode = "segments";

    /* ================= SONIDOS GUÍA (WEB AUDIO API) ================= */

    function ensureAudioContext() {
        if (!soundEnabled) return null;
        const AudioCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtor) return null;
        if (!audioCtx) {
            audioCtx = new AudioCtor();
        }
        if (audioCtx.state === "suspended") {
            audioCtx.resume().catch(() => {});
        }
        return audioCtx;
    }

    function playTone(freq, durationSec, volume = 0.08, type = "sine", delaySec = 0) {
        const ctx = ensureAudioContext();
        if (!ctx) return;
        try {
            const now = ctx.currentTime + delaySec;
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(freq, now);

            gain.gain.setValueAtTime(0.0001, now);
            gain.gain.exponentialRampToValueAtTime(volume, now + 0.015);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + durationSec);

            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + durationSec + 0.02);
        } catch {
            // Ignorar si el navegador restringe Web Audio
        }
    }

    function playCountdownBeep(urgent = true) {
        playTone(urgent ? 880 : 660, 0.11, urgent ? 0.09 : 0.05, "sine");
    }

    function playShutterSound() {
        playTone(1180, 0.07, 0.1, "triangle", 0);
        playTone(620, 0.09, 0.08, "sine", 0.045);
    }

    function playChime() {
        playTone(523.25, 0.14, 0.06, "sine", 0);
        playTone(659.25, 0.18, 0.06, "sine", 0.09);
    }

    function updateSoundButton() {
        const btn = $("auto-sound-btn");
        if (!btn) return;
        btn.textContent = soundEnabled ? "🔊 Sonido" : "🔇 Silencio";
        btn.classList.toggle("active", soundEnabled);
    }

    /* ================= FULLSCREEN Y CÁMARA ================= */

    function updateFullscreenButton() {
        const btn = $("auto-fullscreen-btn");
        if (!btn) return;
        const inFs = Boolean(document.fullscreenElement);
        btn.textContent = inFs ? "↙ Ventana" : "⛶ Pantalla";
        btn.classList.toggle("active", inFs);
    }

    async function requestBrowserFullscreen() {
        try {
            if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
                await document.documentElement.requestFullscreen();
            }
        } catch {
            // Si el navegador bloquea Fullscreen API, #auto-stage sigue cubriendo el 100% del viewport
        } finally {
            updateFullscreenButton();
        }
    }

    async function exitBrowserFullscreen() {
        try {
            if (document.fullscreenElement && document.exitFullscreen) {
                await document.exitFullscreen();
            }
        } catch {
            // Ignorar si ya salió de pantalla completa
        } finally {
            updateFullscreenButton();
        }
    }

    function applyAutoVideoTransform() {
        const video = $("auto-video");
        if (!video) return;
        const { rotation, mirror } = bridge.getCameraSettings();
        const swapped = rotation === 90 || rotation === 270;

        video.style.width = swapped ? "100vh" : "100%";
        video.style.height = swapped ? "100vw" : "100%";

        const parts = [];
        if (mirror) parts.push("scaleX(-1)");
        if (rotation !== 0) parts.push(`rotate(${rotation}deg)`);
        video.style.transform = parts.join(" ");

        const rotBtn = $("auto-rotate-btn");
        const mirBtn = $("auto-mirror-btn");
        if (rotBtn) rotBtn.textContent = `Rotar ${rotation}°`;
        if (mirBtn) {
            mirBtn.textContent = `Espejo: ${mirror ? "Sí" : "No"}`;
            mirBtn.classList.toggle("active", mirror);
        }
    }

    function syncCameraSelect() {
        const mainSelect = $("camera-select");
        const autoSelect = $("auto-camera-select");
        if (!mainSelect || !autoSelect) return;

        autoSelect.innerHTML = "";
        Array.from(mainSelect.options).forEach((opt) => {
            const clone = document.createElement("option");
            clone.value = opt.value;
            clone.textContent = opt.textContent;
            autoSelect.appendChild(clone);
        });
        autoSelect.value = mainSelect.value;
    }

    async function attachCameraToAutoStage() {
        const placeholder = $("auto-video-placeholder");
        const video = $("auto-video");
        if (placeholder) placeholder.hidden = false;

        const stream = await bridge.ensureCameraStarted();
        if (!stream || !isOpen) {
            if (placeholder) placeholder.hidden = true;
            return false;
        }

        video.srcObject = stream;
        video.onloadedmetadata = () => applyAutoVideoTransform();
        applyAutoVideoTransform();
        try {
            await video.play();
        } catch {
            // Algunos navegadores ignoran play() si ya estaba reproduciendo
        }
        if (placeholder) placeholder.hidden = true;
        syncCameraSelect();
        return true;
    }

    function triggerShutterFlash() {
        const flash = $("auto-flash");
        if (!flash) return;
        flash.classList.add("active");
        requestAnimationFrame(() => {
            setTimeout(() => flash.classList.remove("active"), 80);
        });
    }

    /* ================= UTILIDADES DE FASES Y TEMPORIZADOR ================= */

    function hideAllPhaseLayers() {
        $("auto-reference-dock").hidden = true;
        $("auto-timer-dock").hidden = true;
        $("auto-idle-bar").hidden = true;
        $("auto-swap-overlay").hidden = true;
        $("auto-analyzing-overlay").hidden = true;
        $("auto-results-overlay").hidden = true;
        $("auto-final-overlay").hidden = true;
    }

    function isAnyDetailModalOpen() {
        return Boolean(
            $("detail-report-modal")?.classList.contains("open") ||
            bridge.isInspectorOrLightboxOpen()
        );
    }

    function runCountdown({
        totalMs,
        runToken,
        isPaused = () => false,
        isSkipped = () => false,
        onProgress = () => {},
        onSecondChange = () => {}
    }) {
        return new Promise((resolve) => {
            let remaining = totalMs;
            let lastTs = performance.now();
            let lastCeilSec = Math.ceil(remaining / 1000);

            onProgress(remaining, totalMs, isPaused());
            onSecondChange(lastCeilSec);

            const intervalId = setInterval(() => {
                if (runToken !== activeRunToken) {
                    clearInterval(intervalId);
                    resolve({ cancelled: true });
                    return;
                }

                if (isSkipped()) {
                    clearInterval(intervalId);
                    resolve({ skipped: true });
                    return;
                }

                const now = performance.now();
                const paused = isPaused();
                if (!paused) {
                    remaining = Math.max(0, remaining - (now - lastTs));
                }
                lastTs = now;

                onProgress(remaining, totalMs, paused);

                const ceilSec = Math.ceil(remaining / 1000);
                if (ceilSec !== lastCeilSec && ceilSec > 0) {
                    lastCeilSec = ceilSec;
                    onSecondChange(ceilSec);
                }

                if (remaining <= 0) {
                    clearInterval(intervalId);
                    resolve({ completed: true });
                }
            }, 80);
        });
    }

    /* ================= RENDER DE HUD EN PANTALLA COMPLETA ================= */

    function shortSculptureName(sculpture) {
        if (!sculpture) return "Obra";
        return sculpture.title
            .replace("Sebastián de ", "")
            .replace("William ", "");
    }

    function updateTopBar(roundNum, totalRounds, sculpture, player, roleText = "Turno actual") {
        const game = bridge.getGameState();
        const isFree = game.gameMode === "classify";

        $("auto-round-pill").textContent = `Ronda ${roundNum} / ${totalRounds}`;
        $("auto-sculpture-pill").textContent = isFree
            ? "Modo Libre · Elige en secreto"
            : sculpture.title;

        $("auto-player-role").textContent = roleText;
        $("auto-player-name").textContent = player ? player.name : "Preparado";
        $("auto-player-avatar").textContent = (player?.name || "M").trim().charAt(0).toUpperCase();
    }

    function updateReferenceDock(sculpture) {
        const game = bridge.getGameState();
        const isFree = game.gameMode === "classify";

        $("auto-reference-dock").hidden = false;
        $("auto-ref-eyebrow").textContent = isFree ? "Elige en secreto" : "Imita esta obra";
        $("auto-ref-title").textContent = isFree ? "3 esculturas posibles" : sculpture.title;

        $("auto-ref-challenge").hidden = isFree;
        $("auto-ref-free").hidden = !isFree;

        if (isFree) {
            const host = $("auto-ref-free");
            host.innerHTML = "";
            bridge.sculptures.forEach((sc) => {
                const item = document.createElement("div");
                item.className = "auto-ref-free-item";
                const img = document.createElement("img");
                window.setSculptureImage(img, sc);
                const label = document.createElement("span");
                label.textContent = sc.title;
                item.append(img, label);
                host.appendChild(item);
            });
        } else {
            window.setSculptureImage($("auto-ref-img"), sculpture);
            $("auto-ref-tip").textContent = sculpture.tips?.[0] || sculpture.subtitle || "";
        }
    }

    function updatePoseTimerVisual(remainingMs, totalMs) {
        const secs = Math.max(0, Math.ceil(remainingMs / 1000));
        const ratio = Math.max(0, Math.min(1, remainingMs / totalMs));
        const ring = $("auto-timer-ring");
        const dock = $("auto-timer-dock");

        $("auto-timer-number").textContent = String(secs);
        if (ring) {
            ring.style.strokeDasharray = String(RING_CIRCUMFERENCE);
            ring.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - ratio));
        }

        const isUrgent = secs <= 3 && secs > 0;
        dock.classList.toggle("urgent", isUrgent);
        $("auto-timer-label").textContent = isUrgent ? "¡Mantén la pose!" : "Acomódate en pose";
    }

    function updateIdleSummary() {
        const game = bridge.getGameState();
        const names = game.players.map((p) => p.name).join(" · ");
        $("auto-idle-title").textContent = game.gameMode === "classify"
            ? "Modo automático · 3 rondas (Modo Libre)"
            : `Modo automático · 3 esculturas (${bridge.sculptures.length} obras)`;
        $("auto-idle-players").textContent = `Participantes: ${names}`;
        updateTopBar(1, bridge.sculptures.length || 3, game.sculpture, game.players[0], "Listo para iniciar");
    }

    /* ================= EVALUACIÓN EN SEGUNDO PLANO ================= */

    async function evaluateTurnAsync(snapshot, playerIndex, player, sculpture, roundNumber, gameMode) {
        if (!snapshot || !snapshot.blob) {
            return {
                playerIndex,
                playerName: player.name,
                imageUrl: "",
                analysis: null,
                errorMessage: "No se pudo capturar la imagen desde la cámara.",
                score: 0
            };
        }

        try {
            const form = new FormData();
            form.append("image", snapshot.blob, `auto_r${roundNumber}_p${playerIndex + 1}.jpg`);
            form.append("player", player.name);
            form.append("sculpture_id", sculpture.id);
            form.append("sculpture", JSON.stringify(sculpture));
            form.append("round", String(roundNumber));
            form.append("game_mode", gameMode);
            form.append("include_debug_views", "true");

            const response = await fetch(bridge.ANALYSIS_ENDPOINT, {
                method: "POST",
                body: form
            });

            const payload = await response.json().catch(() => null);
            if (!response.ok) {
                return {
                    playerIndex,
                    playerName: player.name,
                    imageUrl: snapshot.dataUrl,
                    analysis: null,
                    errorMessage: payload?.detail || `Error HTTP ${response.status} al evaluar la postura.`,
                    score: 0
                };
            }

            const rawScore = Number(payload?.score);
            const effectiveScore =
                payload?.is_recognized === false || !Number.isFinite(rawScore)
                    ? 0
                    : rawScore;

            return {
                playerIndex,
                playerName: player.name,
                imageUrl: snapshot.dataUrl,
                analysis: payload,
                errorMessage: null,
                score: effectiveScore
            };
        } catch (err) {
            return {
                playerIndex,
                playerName: player.name,
                imageUrl: snapshot.dataUrl,
                analysis: null,
                errorMessage: `No se pudo conectar con Python (${err.message}).`,
                score: 0
            };
        }
    }

    /* ================= RESULTADOS DE RONDA (8 SEGUNDOS) ================= */

    function updateResultsTimerStatusUI() {
        const pausedByModal = isAnyDetailModalOpen();
        const effectivelyPaused = resultsUserPaused || pausedByModal;
        const secs = Math.max(0, Math.ceil(resultsRemainingMs / 1000));
        const pct = Math.max(0, Math.min(100, (resultsRemainingMs / (RESULTS_SECONDS * 1000)) * 100));

        const pauseBtn = $("auto-results-pause-btn");
        const stateTag = $("auto-results-state-tag");
        const nextLabel = $("auto-results-next-label");
        const bar = $("auto-results-bar");

        if (pauseBtn) {
            pauseBtn.textContent = resultsUserPaused ? "▶ Continuar" : "⏸ Pausar";
        }
        if (bar) {
            bar.style.width = `${pct}%`;
        }
        if (stateTag) {
            stateTag.classList.toggle("paused", effectivelyPaused);
            if (resultsUserPaused) {
                stateTag.textContent = "Pausado";
            } else if (pausedByModal) {
                stateTag.textContent = "En pausa · Informe abierto";
            } else {
                stateTag.textContent = "Automático";
            }
        }
        if (nextLabel) {
            if (resultsUserPaused) {
                nextLabel.textContent = `Pausado (${secs}s restantes) · ${resultsNextLabelBase}`;
            } else if (pausedByModal) {
                nextLabel.textContent = `Tiempo detenido (${secs}s) mientras revisas el informe`;
            } else {
                nextLabel.textContent = `${resultsNextLabelBase} en ${secs}s…`;
            }
        }
    }

    function renderRoundResultsBoard(roundRecord, nextSculpture) {
        const game = bridge.getGameState();
        const totalRounds = bridge.sculptures.length || 3;

        $("auto-results-eyebrow").textContent = `Resultados · Ronda ${roundRecord.roundNumber} de ${totalRounds}`;
        $("auto-results-title").textContent = roundRecord.isFreeMode
            ? `Ronda ${roundRecord.roundNumber} · Modo Libre (Adivina la Obra)`
            : roundRecord.sculpture.title;

        // Ranking acumulado compacto en el encabezado
        const standingsHost = $("auto-results-standings");
        standingsHost.innerHTML = "";
        const sortedPlayers = [...game.players].sort((a, b) => b.totalScore - a.totalScore);
        sortedPlayers.forEach((p, idx) => {
            const chip = document.createElement("div");
            chip.className = `auto-standing-chip${idx === 0 ? " leader" : ""}`;
            chip.innerHTML = `<span>#${idx + 1} ${escapeHtml(p.name)}</span><strong>${Math.round(p.totalScore)} pts</strong>`;
            standingsHost.appendChild(chip);
        });

        // Calcular orden de la ronda para insignias #1, #2...
        const scoresDesc = [...roundRecord.turns]
            .map((t, i) => ({ i, score: t.score }))
            .sort((a, b) => b.score - a.score);
        const rankByIndex = new Map();
        scoresDesc.forEach((entry, rankIdx) => rankByIndex.set(entry.i, rankIdx + 1));
        const topRoundScore = scoresDesc[0]?.score ?? 0;

        // Tarjetas comparativas por jugador
        const grid = $("auto-results-grid");
        grid.innerHTML = "";

        roundRecord.turns.forEach((turn, idx) => {
            const rank = rankByIndex.get(idx) || idx + 1;
            const isRoundWinner = turn.score > 0 && turn.score === topRoundScore;
            const tier = bridge.scoreTier(turn.analysis?.is_recognized === false ? 0 : turn.score);

            const card = document.createElement("article");
            card.className = `auto-player-result-card${isRoundWinner ? " round-winner" : ""}`;

            // Cabecera de la tarjeta
            const top = document.createElement("div");
            top.className = "auto-card-top";
            top.innerHTML = `
                <div class="auto-card-player">
                    <span class="avatar">${escapeHtml(turn.playerName.trim().charAt(0).toUpperCase())}</span>
                    <strong>${escapeHtml(turn.playerName)}</strong>
                </div>
                <span class="auto-card-rank">#${rank} ronda</span>
            `;
            card.appendChild(top);

            // Visualización de la foto con esqueleto dibujado
            const visual = document.createElement("div");
            visual.className = "auto-card-visual";
            if (turn.imageUrl && Array.isArray(turn.analysis?.landmarks) && turn.analysis.landmarks.length) {
                bridge.drawPoseOverlay(visual, turn.imageUrl, turn.analysis.landmarks, null);
            } else if (turn.imageUrl) {
                const img = document.createElement("img");
                img.src = turn.imageUrl;
                img.alt = `Captura de ${turn.playerName}`;
                visual.appendChild(img);
            } else {
                visual.innerHTML = '<div class="visual-placeholder">Sin captura disponible</div>';
            }
            card.appendChild(visual);

            // Puntaje + feedback o error
            const scoreRow = document.createElement("div");
            scoreRow.className = "auto-card-score-row";

            const ring = document.createElement("div");
            ring.className = `score-ring ${turn.errorMessage || turn.analysis?.is_recognized === false ? "low" : tier}`;
            ring.textContent = turn.errorMessage ? "0" : `${Math.round(turn.score)}`;

            const fb = document.createElement("div");
            fb.className = "auto-card-feedback";
            const fbTitle = document.createElement("strong");
            const fbDesc = document.createElement("p");

            if (turn.errorMessage) {
                fbTitle.textContent = "Sin detección válida (0 pts)";
                fbDesc.textContent = turn.errorMessage;
            } else if (turn.analysis?.is_recognized === false) {
                fbTitle.textContent = "Postura no reconocida (0 pts)";
                fbDesc.textContent = turn.analysis.feedback || "Intenta marcar más la silueta en la siguiente ronda.";
            } else if (roundRecord.isFreeMode) {
                const matched = bridge.sculptures.find((s) => s.id === turn.analysis?.sculpture_id);
                fbTitle.textContent = matched ? `Detectada: ${shortSculptureName(matched)}` : "Evaluación lista";
                fbDesc.textContent = turn.analysis?.feedback || "Evaluación completada.";
            } else {
                fbTitle.textContent =
                    tier === "high" ? "Imitación excelente"
                    : tier === "mid" ? "Muy cerca de la obra"
                    : "Ajusta tu postura";
                fbDesc.textContent = turn.analysis?.feedback || "Evaluación completada.";
            }

            fb.append(fbTitle, fbDesc);
            scoreRow.append(ring, fb);
            card.appendChild(scoreRow);

            // Error destacado o mini mapa de calor por región
            if (turn.errorMessage) {
                const errBox = document.createElement("div");
                errBox.className = "auto-card-error";
                errBox.textContent = turn.errorMessage;
                card.appendChild(errBox);
            } else if (turn.analysis?.regions) {
                const regWrap = document.createElement("div");
                regWrap.className = "auto-card-regions";
                const labels = { head: "Cabeza", torso: "Torso", arms: "Brazos", legs: "Piernas" };
                Object.entries(labels).forEach(([key, label]) => {
                    const raw = turn.analysis.regions[key];
                    const val = typeof raw === "number" ? raw : raw?.similarity ?? null;
                    const regTier = bridge.scoreTier(val);
                    const cell = document.createElement("div");
                    cell.className = "auto-mini-region";
                    cell.innerHTML = `
                        <div class="auto-mini-region-head">
                            <span>${label}</span>
                            <strong>${val === null ? "—" : `${Math.round(val)}%`}</strong>
                        </div>
                        <div class="heat-track">
                            <div class="heat-fill ${regTier}" style="width:${val === null ? 0 : Math.max(0, Math.min(100, val))}%"></div>
                        </div>
                    `;
                    regWrap.appendChild(cell);
                });
                card.appendChild(regWrap);
            }

            // Botón directo para abrir el informe detallado en la pestaña de este jugador
            const detailBtn = document.createElement("button");
            detailBtn.type = "button";
            detailBtn.className = "btn btn-ghost btn-sm auto-card-detail-btn";
            detailBtn.textContent = "Ver informe detallado";
            detailBtn.addEventListener("click", () => openDetailReportModal(roundRecord, idx));
            card.appendChild(detailBtn);

            grid.appendChild(card);
        });

        // Etiqueta del siguiente paso en el footer
        if (nextSculpture) {
            resultsNextLabelBase = roundRecord.isFreeMode
                ? `Siguiente ronda (${roundRecord.roundNumber + 1} de ${totalRounds})`
                : `Siguiente obra: ${nextSculpture.title}`;
            $("auto-results-next-btn").textContent = "Siguiente obra →";
        } else {
            resultsNextLabelBase = "Podio final";
            $("auto-results-next-btn").textContent = "Ver podio final →";
        }

        // Botón general de informe detallado abre en el primer jugador
        $("auto-results-detail-btn").onclick = () => openDetailReportModal(roundRecord, 0);
    }

    /* ================= MODAL DE INFORME DETALLADO ================= */

    function renderDetailPlayerStage(turn) {
        const container = $("detail-player-stage");
        if (!container) return;

        const data = turn.analysis;
        const baseImage = data?.annotated_image || turn.imageUrl;
        const sculptView = data?.debug_views?.sculpture_mode_view;

        if (!baseImage) {
            container.innerHTML = '<div class="visual-placeholder">No hay fotografía capturada para este turno.</div>';
            return;
        }

        if (detailViewMode === "sculpture" && sculptView) {
            container.innerHTML = "";
            const img = document.createElement("img");
            img.src = sculptView;
            img.alt = `Silueta en Modo Escultura de ${turn.playerName}`;
            container.appendChild(img);
            return;
        }

        if (Array.isArray(data?.landmarks) && data.landmarks.length) {
            const heatmapArg = detailViewMode === "heatmap" ? (data.regions || null) : null;
            bridge.drawPoseOverlay(container, baseImage, data.landmarks, heatmapArg);
        } else {
            container.innerHTML = "";
            const img = document.createElement("img");
            img.src = baseImage;
            img.alt = `Captura de ${turn.playerName}`;
            container.appendChild(img);
        }
    }

    function renderDetailReportContent() {
        if (!detailRoundRecord) return;
        const turn = detailRoundRecord.turns[detailPlayerIndex] || detailRoundRecord.turns[0];
        if (!turn) return;

        const totalRounds = bridge.sculptures.length || 3;
        $("detail-eyebrow").textContent = `Informe detallado · Ronda ${detailRoundRecord.roundNumber} de ${totalRounds}`;
        $("detail-title").textContent = detailRoundRecord.isFreeMode
            ? `${turn.playerName} · Modo Libre`
            : `${turn.playerName} · ${detailRoundRecord.sculpture.title}`;
        $("detail-subtitle").textContent =
            "Métricas completas de postura, mapa de calor, clasificación ML y diagnóstico del Pipeline PDI.";

        // Pestañas por jugador
        const tabsHost = $("detail-player-tabs");
        tabsHost.innerHTML = "";
        detailRoundRecord.turns.forEach((t, idx) => {
            const tab = document.createElement("button");
            tab.type = "button";
            tab.role = "tab";
            tab.className = `detail-player-tab${idx === detailPlayerIndex ? " active" : ""}`;
            tab.setAttribute("aria-selected", idx === detailPlayerIndex ? "true" : "false");
            tab.innerHTML = `
                <span>${escapeHtml(t.playerName)}</span>
                <span class="tab-score">${t.errorMessage ? "0 pts" : `${Math.round(t.score)} pts`}</span>
            `;
            tab.addEventListener("click", () => {
                detailPlayerIndex = idx;
                detailViewMode = "segments";
                renderDetailReportContent();
            });
            tabsHost.appendChild(tab);
        });

        const data = turn.analysis;
        const errBanner = $("detail-error-banner");
        if (turn.errorMessage) {
            errBanner.hidden = false;
            errBanner.textContent = `No se completó el análisis automático para ${turn.playerName}: ${turn.errorMessage}`;
        } else {
            errBanner.hidden = true;
            errBanner.textContent = "";
        }

        // Botones de vista de pose
        const hasSculptureView = Boolean(data?.debug_views?.sculpture_mode_view);
        if (detailViewMode === "sculpture" && !hasSculptureView) {
            detailViewMode = "segments";
        }
        $("detail-sculpture-view-btn").disabled = !hasSculptureView;
        document.querySelectorAll(".detail-view-pill").forEach((pill) => {
            pill.classList.toggle("active", pill.dataset.view === detailViewMode);
        });

        // Vista del jugador
        renderDetailPlayerStage(turn);

        // Escultura de referencia + esqueleto base
        const matchedSculpture =
            bridge.sculptures.find((s) => s.id === data?.sculpture_id) || detailRoundRecord.sculpture;
        const refStage = $("detail-ref-stage");

        if (Array.isArray(data?.reference_landmarks) && data.reference_landmarks.length) {
            bridge.drawPoseOverlay(refStage, matchedSculpture.image, data.reference_landmarks);
        } else {
            refStage.innerHTML = '<img alt="Escultura de referencia">';
            window.setSculptureImage(refStage.querySelector("img"), matchedSculpture);
        }

        if (detailRoundRecord.isFreeMode) {
            $("detail-ref-stage-title").textContent = !data
                ? "Obra de referencia"
                : data.is_recognized === false
                ? "Obra más cercana (baja confianza)"
                : `Obra identificada · ${matchedSculpture.title}`;
        } else {
            $("detail-ref-stage-title").textContent = "Escultura de referencia · Esqueleto base";
        }

        // Puntaje
        const scoreEl = $("detail-score-value");
        const titleEl = $("detail-score-title");
        const descEl = $("detail-score-description");

        if (turn.errorMessage || !data) {
            scoreEl.textContent = "0";
            scoreEl.className = "score-ring low";
            titleEl.textContent = "Postura no evaluada";
            descEl.textContent = turn.errorMessage || "No se obtuvieron resultados del servidor Python.";
        } else {
            const tier = bridge.scoreTier(data.is_recognized === false ? 0 : turn.score);
            scoreEl.textContent = data.is_recognized === false ? "—" : `${Math.round(turn.score)}`;
            scoreEl.className = `score-ring ${data.is_recognized === false ? "low" : tier}`;

            if (data.is_recognized === false) {
                titleEl.textContent = "Postura no reconocida";
            } else if (detailRoundRecord.isFreeMode) {
                titleEl.textContent = `Obra identificada: ${matchedSculpture.title}`;
            } else {
                titleEl.textContent =
                    tier === "high" ? "Imitación excelente"
                    : tier === "mid" ? "Muy cerca de la obra"
                    : "Ajusta tu postura";
            }
            descEl.textContent = data.feedback || "Evaluación completada.";
        }

        // Clasificador ML + Mapa de calor
        bridge.renderClassifierProbsTo(
            $("detail-classifier-prediction"),
            $("detail-classifier-probs"),
            data || null
        );
        bridge.renderHeatmapTo($("detail-heatmap-regions"), data?.regions || null);

        // Diagnóstico PDI y Segmentación
        const pdi = data?.pdi_metrics || null;
        const sil = data?.silhouette_metrics || null;
        $("detail-pdi-sharpness").textContent = pdi
            ? `${Math.round(pdi.laplacian_var)} (${pdi.is_blurry ? "Borrosa" : "Nítida"})`
            : "—";
        $("detail-pdi-clahe").textContent = pdi
            ? (pdi.clahe_applied ? `Sí (L=${Math.round(pdi.mean_luminance)})` : `No (L=${Math.round(pdi.mean_luminance)})`)
            : "—";
        $("detail-pdi-confidence").textContent = pdi && typeof pdi.mean_visibility === "number"
            ? `${Math.round(pdi.mean_visibility * 100)}%`
            : "—";
        $("detail-pdi-iou").textContent = sil && typeof sil.iou === "number"
            ? `${Math.round(sil.iou)}%`
            : "—";
        $("detail-pdi-hu").textContent = sil && typeof sil.hu_similarity === "number"
            ? `${Math.round(sil.hu_similarity)}%`
            : "—";

        // Botón "Ver informe PDI" -> abre el Inspector del Pipeline PDI con Lightbox ampliable
        const pdiBtn = $("detail-open-pdi-btn");
        pdiBtn.disabled = !data?.debug_views;
        pdiBtn.onclick = () => {
            if (!data?.debug_views) return;
            bridge.openInspectorModal(data, turn.imageUrl);
        };

        // Contexto cultural
        $("detail-cultural-description").textContent = detailRoundRecord.isFreeMode && data
            ? `${matchedSculpture.title} (${matchedSculpture.author} · ${matchedSculpture.emotion}): ${matchedSculpture.description}`
            : matchedSculpture.description;
    }

    function openDetailReportModal(roundRecord, playerIdx = 0) {
        if (!roundRecord) return;
        detailRoundRecord = roundRecord;
        detailPlayerIndex = Math.max(0, Math.min(roundRecord.turns.length - 1, playerIdx));
        detailViewMode = "segments";
        renderDetailReportContent();
        $("detail-report-modal").classList.add("open");
        updateResultsTimerStatusUI();
    }

    function closeDetailReportModal() {
        const modal = $("detail-report-modal");
        if (!modal?.classList.contains("open")) return;
        modal.classList.remove("open");
        updateResultsTimerStatusUI();
    }

    /* ================= PODIO FINAL ================= */

    function renderFinalPodium() {
        const game = bridge.getGameState();
        const sorted = [...game.players]
            .map((p, origIdx) => ({ player: p, origIdx }))
            .sort((a, b) => b.player.totalScore - a.player.totalScore);

        const topScore = sorted[0]?.player.totalScore ?? 0;
        const winners = sorted.filter((x) => Math.round(x.player.totalScore) === Math.round(topScore));

        if (winners.length > 1) {
            $("auto-final-winner").textContent = `¡Empate en el primer lugar!`;
            $("auto-final-subtitle").textContent =
                `${winners.map((w) => w.player.name).join(" y ")} lideran con ${Math.round(topScore)} puntos acumulados tras las ${sessionHistory.length} rondas.`;
        } else if (winners[0]) {
            $("auto-final-winner").textContent = `¡Ganador: ${winners[0].player.name}!`;
            $("auto-final-subtitle").textContent =
                `Completaron las ${sessionHistory.length} esculturas con un acumulado de ${Math.round(topScore)} puntos.`;
        } else {
            $("auto-final-winner").textContent = "¡Partida completada!";
            $("auto-final-subtitle").textContent = "";
        }

        // Encabezado dinámico con los nombres de las 3 obras
        const thead = $("auto-final-thead");
        const roundCols = sessionHistory
            .map((r) =>
                r.isFreeMode
                    ? `<th class="num">Ronda ${r.roundNumber}</th>`
                    : `<th class="num">${escapeHtml(shortSculptureName(r.sculpture))}</th>`
            )
            .join("");
        thead.innerHTML = `
            <tr>
                <th>Pos.</th>
                <th>Participante</th>
                ${roundCols}
                <th class="num">Total</th>
            </tr>
        `;

        const tbody = $("auto-final-tbody");
        tbody.innerHTML = "";
        sorted.forEach(({ player, origIdx }, rankIdx) => {
            const perRoundCells = sessionHistory
                .map((r) => {
                    const turn = r.turns.find((t) => t.playerIndex === origIdx);
                    return `<td class="num">${turn ? Math.round(turn.score) : "—"}</td>`;
                })
                .join("");

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>#${rankIdx + 1}</td>
                <td><strong>${escapeHtml(player.name)}</strong></td>
                ${perRoundCells}
                <td class="num"><strong>${Math.round(player.totalScore)}</strong></td>
            `;
            tbody.appendChild(tr);
        });

        // Botones rápidos para consultar el informe detallado de cualquiera de las 3 rondas
        const roundsHost = $("auto-final-rounds");
        roundsHost.innerHTML = "";
        if (sessionHistory.length) {
            const label = document.createElement("span");
            label.className = "auto-final-rounds-label";
            label.textContent = "Revisar informes:";
            roundsHost.appendChild(label);

            sessionHistory.forEach((rec) => {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "btn btn-ghost btn-sm";
                btn.textContent = rec.isFreeMode
                    ? `Ronda ${rec.roundNumber}`
                    : `R${rec.roundNumber} · ${shortSculptureName(rec.sculpture)}`;
                btn.addEventListener("click", () => openDetailReportModal(rec, 0));
                roundsHost.appendChild(btn);
            });
        }
    }

    /* ================= BUCLE PRINCIPAL AUTOMÁTICO (3 OBRAS) ================= */

    async function startAutomaticMatch() {
        if (!isOpen) return;

        // Preparar audio bajo gesto explícito del usuario
        ensureAudioContext();

        const hasCamera = bridge.getStream() ? true : await attachCameraToAutoStage();
        if (!hasCamera) {
            bridge.notify("Activa una cámara válida para iniciar el modo automático.");
            return;
        }

        const runToken = ++activeRunToken;
        isRunning = true;
        sessionHistory = [];

        const game = bridge.getGameState();
        const sculptures = bridge.sculptures;
        const totalRounds = sculptures.length || 3;
        const startIdx = Math.max(0, sculptures.findIndex((s) => s.id === game.sculpture.id));
        const roundSculptures = Array.from({ length: totalRounds }, (_, i) => sculptures[(startIdx + i) % sculptures.length]);

        // Reiniciar puntajes para una partida completa limpia de 3 esculturas
        game.players.forEach((p) => {
            p.totalScore = 0;
            p.lastScore = null;
            p.roundsPlayed = 0;
        });
        game.round = 1;
        game.currentPlayer = 0;
        game.sculpture = roundSculptures[0];
        game.sculptureId = roundSculptures[0].id;
        bridge.saveState();
        bridge.renderGame();

        const isFreeMode = game.gameMode === "classify";

        for (let r = 0; r < totalRounds; r++) {
            if (runToken !== activeRunToken) return;

            const roundNumber = r + 1;
            const sculpture = roundSculptures[r];
            const nextSculpture = r < totalRounds - 1 ? roundSculptures[r + 1] : null;

            game.round = roundNumber;
            game.sculpture = sculpture;
            game.sculptureId = sculpture.id;
            bridge.saveState();
            bridge.renderGame();

            const roundTurnPromises = [];

            // Turno por cada participante (2 a 4 jugadores)
            for (let p = 0; p < game.players.length; p++) {
                if (runToken !== activeRunToken) return;

                const player = game.players[p];
                game.currentPlayer = p;
                bridge.saveState();
                bridge.renderGame();

                // Fase 1: 10 segundos de preparación en vivo
                hideAllPhaseLayers();
                updateTopBar(roundNumber, totalRounds, sculpture, player, `Turno ${p + 1} de ${game.players.length}`);
                updateReferenceDock(sculpture);
                $("auto-timer-dock").hidden = false;

                const poseResult = await runCountdown({
                    totalMs: POSE_SECONDS * 1000,
                    runToken,
                    onProgress: (rem, total) => updatePoseTimerVisual(rem, total),
                    onSecondChange: (sec) => {
                        if (sec <= 3 && sec >= 1) {
                            playCountdownBeep(true);
                        }
                    }
                });
                if (poseResult.cancelled || runToken !== activeRunToken) return;

                // Captura instantánea + flash + sonido obturador
                playShutterSound();
                triggerShutterFlash();
                const snapshot = await bridge.captureSnapshotFromVideo($("auto-video"));
                await new Promise((res) => setTimeout(res, 180));
                if (runToken !== activeRunToken) return;

                // Evaluar en segundo plano mientras ocurre el cambio de jugador
                const evalPromise = evaluateTurnAsync(
                    snapshot,
                    p,
                    player,
                    sculpture,
                    roundNumber,
                    game.gameMode
                );
                roundTurnPromises.push(evalPromise);

                // Fase 2: Cambio de jugador (5 segundos) si hay siguiente participante
                if (p < game.players.length - 1) {
                    const nextPlayer = game.players[p + 1];
                    hideAllPhaseLayers();
                    updateTopBar(roundNumber, totalRounds, sculpture, nextPlayer, "Cambio de lugar");

                    $("auto-swap-captured").textContent = `✓ Foto de ${player.name} capturada`;
                    $("auto-swap-title").textContent = `Turno de ${nextPlayer.name}`;
                    $("auto-swap-instruction").textContent =
                        `Por favor, ${player.name} sal del encuadre de la foto y pasa ${nextPlayer.name}.`;
                    $("auto-swap-next-name").textContent = nextPlayer.name;
                    $("auto-swap-next-avatar").textContent = nextPlayer.name.trim().charAt(0).toUpperCase();
                    $("auto-swap-overlay").hidden = false;

                    const swapResult = await runCountdown({
                        totalMs: SWAP_SECONDS * 1000,
                        runToken,
                        onProgress: (rem, total) => {
                            const secs = Math.max(0, Math.ceil(rem / 1000));
                            $("auto-swap-seconds").textContent = String(secs);
                            $("auto-swap-bar").style.width = `${Math.max(0, Math.min(100, (rem / total) * 100))}%`;
                        },
                        onSecondChange: (sec) => {
                            if (sec <= 3 && sec >= 1) {
                                playCountdownBeep(false);
                            }
                        }
                    });
                    if (swapResult.cancelled || runToken !== activeRunToken) return;
                }
            }

            // Fase 3: Esperar a que finalicen todas las evaluaciones Python de la ronda
            hideAllPhaseLayers();
            $("auto-analyzing-title").textContent = `Evaluando posturas · Ronda ${roundNumber} de ${totalRounds}`;
            $("auto-analyzing-overlay").hidden = false;

            const turnResults = await Promise.all(roundTurnPromises);
            if (runToken !== activeRunToken) return;

            // Actualizar puntajes acumulados de los jugadores
            turnResults.forEach((turn) => {
                const player = game.players[turn.playerIndex];
                if (!player) return;
                player.lastScore = turn.score;
                player.totalScore += turn.score;
                player.roundsPlayed += 1;
            });
            bridge.saveState();
            bridge.renderGame();

            const roundRecord = {
                roundNumber,
                sculpture,
                isFreeMode,
                turns: turnResults
            };
            sessionHistory.push(roundRecord);

            // Fase 4: Mostrar resultados de la ronda durante 8 segundos (pausable + Informe Detallado)
            hideAllPhaseLayers();
            updateTopBar(roundNumber, totalRounds, sculpture, null, `Cierre de ronda ${roundNumber}`);
            $("auto-player-name").textContent = isFreeMode
                ? `Resultados Ronda ${roundNumber}`
                : shortSculptureName(sculpture);
            $("auto-player-avatar").textContent = String(roundNumber);

            resultsUserPaused = false;
            resultsSkipRequested = false;
            resultsRemainingMs = RESULTS_SECONDS * 1000;

            renderRoundResultsBoard(roundRecord, nextSculpture);
            $("auto-results-overlay").hidden = false;
            updateResultsTimerStatusUI();
            playChime();

            const resCountdown = await runCountdown({
                totalMs: RESULTS_SECONDS * 1000,
                runToken,
                isPaused: () => resultsUserPaused || isAnyDetailModalOpen(),
                isSkipped: () => resultsSkipRequested,
                onProgress: (rem) => {
                    resultsRemainingMs = rem;
                    updateResultsTimerStatusUI();
                }
            });
            if (resCountdown.cancelled || runToken !== activeRunToken) return;

            // Cerrar cualquier modal de informe si el usuario pulsó "Siguiente ahora"
            closeDetailReportModal();
            bridge.closePdiLightbox();
            bridge.closeInspectorModal();
        }

        // Fase 5: Podio final tras haber completado las 3 esculturas
        if (runToken !== activeRunToken) return;
        isRunning = false;
        hideAllPhaseLayers();
        updateTopBar(totalRounds, totalRounds, roundSculptures[totalRounds - 1], null, "Partida terminada");
        $("auto-player-name").textContent = "Podio Final";
        $("auto-player-avatar").textContent = "★";
        renderFinalPodium();
        $("auto-final-overlay").hidden = false;
        playChime();
    }

    /* ================= APERTURA Y CIERRE DEL MODO AUTOMÁTICO ================= */

    async function openAutoMode() {
        if (isOpen) return;
        isOpen = true;
        isRunning = false;
        activeRunToken += 1;

        const stage = $("auto-stage");
        stage.hidden = false;
        document.body.style.overflow = "hidden";

        hideAllPhaseLayers();
        $("auto-idle-bar").hidden = false;
        updateIdleSummary();
        updateSoundButton();
        syncCameraSelect();
        applyAutoVideoTransform();

        // Disparar pantalla completa dentro del gesto de clic sin bloquear el enlace de la cámara
        requestBrowserFullscreen();
        await attachCameraToAutoStage();
    }

    async function closeAutoMode() {
        if (!isOpen) return;
        activeRunToken += 1;
        isOpen = false;
        isRunning = false;
        resultsUserPaused = false;
        resultsSkipRequested = false;

        closeDetailReportModal();
        bridge.closePdiLightbox();
        bridge.closeInspectorModal();

        const video = $("auto-video");
        if (video) video.srcObject = null;

        $("auto-stage").hidden = true;
        document.body.style.overflow = "";

        await exitBrowserFullscreen();
        bridge.renderGame();
    }

    function escapeHtml(str) {
        return String(str ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    /* ================= EVENTOS ================= */

    $("open-auto-mode-btn")?.addEventListener("click", openAutoMode);
    $("auto-exit-btn")?.addEventListener("click", closeAutoMode);
    $("auto-back-normal-btn")?.addEventListener("click", closeAutoMode);
    $("auto-back-lobby-btn")?.addEventListener("click", async () => {
        await closeAutoMode();
        bridge.stopCamera();
        window.location.href = "index.html";
    });

    $("auto-start-btn")?.addEventListener("click", startAutomaticMatch);
    $("auto-play-again-btn")?.addEventListener("click", startAutomaticMatch);

    $("auto-sound-btn")?.addEventListener("click", () => {
        soundEnabled = !soundEnabled;
        updateSoundButton();
        if (soundEnabled) playCountdownBeep(false);
    });

    $("auto-fullscreen-btn")?.addEventListener("click", () => {
        if (document.fullscreenElement) {
            exitBrowserFullscreen();
        } else {
            requestBrowserFullscreen();
        }
    });

    $("auto-rotate-btn")?.addEventListener("click", () => {
        bridge.rotateCamera();
        applyAutoVideoTransform();
    });

    $("auto-mirror-btn")?.addEventListener("click", () => {
        bridge.toggleMirror();
        applyAutoVideoTransform();
    });

    $("auto-camera-select")?.addEventListener("change", async (e) => {
        const mainSelect = $("camera-select");
        if (mainSelect) mainSelect.value = e.target.value;
        await bridge.startCamera(e.target.value);
        if (isOpen) {
            await attachCameraToAutoStage();
        }
    });

    // Controles de la fase de resultados (8s)
    $("auto-results-pause-btn")?.addEventListener("click", () => {
        resultsUserPaused = !resultsUserPaused;
        updateResultsTimerStatusUI();
    });

    $("auto-results-next-btn")?.addEventListener("click", () => {
        resultsSkipRequested = true;
    });

    // Modal Informe Detallado
    $("detail-close-btn")?.addEventListener("click", closeDetailReportModal);
    $("detail-close-x")?.addEventListener("click", closeDetailReportModal);
    $("detail-report-modal")?.addEventListener("click", (e) => {
        if (e.target === $("detail-report-modal")) {
            closeDetailReportModal();
        }
    });

    document.querySelectorAll(".detail-view-pill").forEach((pill) => {
        pill.addEventListener("click", () => {
            if (pill.disabled) return;
            detailViewMode = pill.dataset.view || "segments";
            document.querySelectorAll(".detail-view-pill").forEach((b) => {
                b.classList.toggle("active", b.dataset.view === detailViewMode);
            });
            const turn = detailRoundRecord?.turns?.[detailPlayerIndex];
            if (turn) renderDetailPlayerStage(turn);
        });
    });

    // Tecla Escape para cerrar el Informe Detallado (si no están abiertos Lightbox ni Inspector PDI)
    document.addEventListener("keydown", (e) => {
        if (e.key !== "Escape") return;
        if (bridge.isInspectorOrLightboxOpen()) return;
        if ($("detail-report-modal")?.classList.contains("open")) {
            e.preventDefault();
            closeDetailReportModal();
        }
    });

    // Reaccionar al abrir/cerrar Inspector PDI o Lightbox (mantiene pausado el reloj de resultados)
    document.addEventListener("mimesis:modalchange", () => {
        if (isOpen && !$("auto-results-overlay").hidden) {
            updateResultsTimerStatusUI();
        }
    });

    document.addEventListener("mimesis:camerasupdated", syncCameraSelect);
    document.addEventListener("fullscreenchange", updateFullscreenButton);

    updateSoundButton();
    updateFullscreenButton();
})();
