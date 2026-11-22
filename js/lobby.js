/* ==========================================================
   lobby.js
   Lógica de la pantalla de configuración (index.html)
   ========================================================== */

(() => {
    "use strict";

    const sculptures = window.MIMESIS_SCULPTURES || [];
    const $ = (id) => document.getElementById(id);

    let players = [
        { name: "Jugador 1" },
        { name: "Jugador 2" }
    ];
    let selectedIndex = 0;
    let gameMode = "challenge";
    let toastTimer = null;

    function notify(message) {
        const el = $("notification");
        el.textContent = message;
        el.classList.add("show");
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => el.classList.remove("show"), 2800);
    }

    /* Comprueba si el servidor Python (FastAPI) está respondiendo */
    async function checkBackendStatus() {
        const dot = $("backend-dot");
        const label = $("backend-label");
        if (!dot || !label) return;
        try {
            const res = await fetch("/api/health", { cache: "no-store" });
            if (!res.ok) throw new Error();
            const data = await res.json();
            dot.classList.remove("offline");
            dot.classList.add("online");
            label.textContent = data.mlp_loaded
                ? "Motor Python · MLP activo"
                : "Motor Python · Modo geométrico";
        } catch {
            dot.classList.remove("online");
            dot.classList.add("offline");
            label.textContent = "Motor Python desconectado";
        }
    }

    function renderPlayers() {
        const list = $("players-list");
        list.innerHTML = "";

        players.forEach((player, idx) => {
            const row = document.createElement("div");
            row.className = "player-item";

            const avatar = document.createElement("div");
            avatar.className = "avatar";
            avatar.textContent = (player.name || "J").trim().charAt(0).toUpperCase();

            const input = document.createElement("input");
            input.className = "player-input";
            input.type = "text";
            input.maxLength = 22;
            input.value = player.name;
            input.setAttribute("aria-label", `Nombre del jugador ${idx + 1}`);
            input.addEventListener("input", (e) => {
                players[idx].name = e.target.value;
                avatar.textContent = (e.target.value || "J").trim().charAt(0).toUpperCase();
            });

            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "remove-player";
            remove.textContent = "×";
            remove.title = `Eliminar a ${player.name}`;
            remove.addEventListener("click", () => {
                if (players.length <= 2) {
                    notify("La partida requiere mínimo 2 participantes.");
                    return;
                }
                players.splice(idx, 1);
                renderPlayers();
            });

            row.append(avatar, input, remove);
            list.appendChild(row);
        });

        $("player-counter").textContent = `${players.length} / 4`;
        $("add-player").disabled = players.length >= 4;

        document.querySelectorAll(".count-option").forEach((btn) => {
            btn.classList.toggle("active", Number(btn.dataset.count) === players.length);
        });
    }

    function selectSculpture(index) {
        selectedIndex = index;
        const s = sculptures[index];
        if (!s) return;

        window.setSculptureImage($("main-image"), s);
        $("main-title").textContent = s.title;
        $("main-subtitle").textContent = s.subtitle;

        $("info-title").textContent = s.title;
        $("info-author").textContent = s.author;
        $("info-emotion").textContent = s.emotion;
        $("info-description").textContent = s.description;

        const tipsList = $("info-tips");
        tipsList.innerHTML = "";
        (s.tips || []).forEach((tip) => {
            const li = document.createElement("li");
            li.textContent = tip;
            tipsList.appendChild(li);
        });

        renderGallery();
    }

    function renderGallery() {
        const container = $("gallery-list");
        container.innerHTML = "";

        sculptures.forEach((s, index) => {
            const card = document.createElement("article");
            card.className = `gallery-card${index === selectedIndex ? " selected" : ""}`;
            card.tabIndex = 0;

            const thumb = document.createElement("div");
            thumb.className = "gallery-thumb";
            const img = document.createElement("img");
            window.setSculptureImage(img, s);
            thumb.appendChild(img);

            const body = document.createElement("div");
            body.className = "gallery-body";
            const title = document.createElement("strong");
            title.textContent = s.title;
            const author = document.createElement("small");
            author.textContent = `${s.author} · ${s.emotion}`;
            body.append(title, author);

            card.append(thumb, body);
            card.addEventListener("click", () => selectSculpture(index));
            card.addEventListener("keydown", (e) => {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    selectSculpture(index);
                }
            });
            container.appendChild(card);
        });
    }

    /* Eventos */
    $("add-player").addEventListener("click", () => {
        if (players.length >= 4) {
            notify("Máximo 4 participantes por partida.");
            return;
        }
        players.push({ name: `Jugador ${players.length + 1}` });
        renderPlayers();
    });

    document.querySelectorAll(".count-option").forEach((btn) => {
        btn.addEventListener("click", () => {
            const count = Number(btn.dataset.count);
            while (players.length < count) {
                players.push({ name: `Jugador ${players.length + 1}` });
            }
            while (players.length > count) {
                players.pop();
            }
            renderPlayers();
        });
    });

    document.querySelectorAll(".mode-option").forEach((btn) => {
        btn.addEventListener("click", () => {
            gameMode = btn.dataset.mode === "classify" ? "classify" : "challenge";
            document.querySelectorAll(".mode-option").forEach((b) => {
                b.classList.toggle("active", b.dataset.mode === gameMode);
            });
            const helper = $("mode-helper-text");
            if (helper) {
                helper.textContent = gameMode === "classify"
                    ? "Modo Libre: cada jugador elige en secreto una de las 3 esculturas y el clasificador ML la identifica y califica automáticamente."
                    : "Haz clic sobre cualquier nombre para editarlo. Cada jugador tendrá su turno y al cerrar la ronda se actualizará el ranking.";
            }
        });
    });

    $("start-game").addEventListener("click", () => {
        const cleanPlayers = players.map((p, i) => ({
            name: (p.name || "").trim() || `Jugador ${i + 1}`,
            totalScore: 0,
            lastScore: null,
            roundsPlayed: 0
        }));

        if (cleanPlayers.length < 2 || cleanPlayers.length > 4) {
            notify("Configura entre 2 y 4 jugadores.");
            return;
        }

        const gameState = {
            players: cleanPlayers,
            gameMode,
            sculptureId: sculptures[selectedIndex].id,
            sculpture: sculptures[selectedIndex],
            currentPlayer: 0,
            round: 1,
            roundScores: []
        };

        sessionStorage.setItem("mimesisGame", JSON.stringify(gameState));
        window.location.href = "game.html";
    });

    /* Inicialización */
    renderPlayers();
    selectSculpture(0);
    checkBackendStatus();
})();
