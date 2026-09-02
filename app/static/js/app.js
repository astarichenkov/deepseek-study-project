/* DeepSeek Study App — comparison frontend (управление ответом модели).
 * Vanilla JavaScript (no framework). Talks to POST /api/compare.
 *
 * One click = exactly ONE /api/compare request; the backend performs
 * exactly TWO DeepSeek provider calls (unrestricted + controlled).
 */
(function () {
  "use strict";

  function initApp() {
    // Required DOM contract. If the markup and JS get out of sync (stale
    // cache / partial deployment), fail CLEARLY in the console instead of
    // throwing somewhere deep and silently disabling every control.
    var requiredIds = [
      "compare-form", "message", "response-format", "max-tokens",
      "stop-sequence", "reset-json", "compare-button", "loading",
      "error", "api-preview", "results", "answer-unrestricted",
      "finish-unrestricted", "answer-controlled", "finish-controlled",
      "applied-settings", "applied-heading", "controlled-error",
      "summary-section", "summary"
    ];
    var missing = requiredIds.filter(function (id) {
      return !document.getElementById(id);
    });
    if (missing.length > 0) {
      console.error(
        "DeepSeek Study App: missing DOM elements, frontend disabled:",
        missing.join(", ")
      );
      return;
    }

    var form = document.getElementById("compare-form");
  var messageEl = document.getElementById("message");
  var formatEl = document.getElementById("response-format");
  var maxTokensEl = document.getElementById("max-tokens");
  var stopEl = document.getElementById("stop-sequence");
  var resetJsonBtn = document.getElementById("reset-json");
  var button = document.getElementById("compare-button");
  var loading = document.getElementById("loading");
  var errorBox = document.getElementById("error");
  var previewEl = document.getElementById("api-preview");

  var resultsSection = document.getElementById("results");
  var answerUnrestricted = document.getElementById("answer-unrestricted");
  var finishUnrestricted = document.getElementById("finish-unrestricted");
  var answerControlled = document.getElementById("answer-controlled");
  var finishControlled = document.getElementById("finish-controlled");
  var appliedSettings = document.getElementById("applied-settings");
  var appliedHeading = document.getElementById("applied-heading");
  var controlledError = document.getElementById("controlled-error");
  var summarySection = document.getElementById("summary-section");
  var summaryList = document.getElementById("summary");

  var DEFAULT_JSON = '{\n  "type": "json_object"\n}';
  var inFlight = false;

  /* ---------------- Tooltips: hover, focus, tap ---------------- */
  document.querySelectorAll(".info-btn").forEach(function (btn) {
    var tip = document.getElementById(btn.getAttribute("aria-controls"));
    if (!tip) return;

    function show() {
      tip.classList.add("open");
      btn.setAttribute("aria-expanded", "true");
    }
    function hide() {
      tip.classList.remove("open");
      btn.setAttribute("aria-expanded", "false");
    }

    btn.addEventListener("mouseenter", show);
    btn.addEventListener("mouseleave", hide);
    btn.addEventListener("focus", show);   // keyboard focus
    btn.addEventListener("blur", hide);
    btn.addEventListener("click", function (event) {
      event.preventDefault();               // touch: toggle
      if (tip.classList.contains("open")) {
        hide();
      } else {
        show();
      }
    });
    btn.addEventListener("keydown", function (event) {
      if (event.key === "Escape") hide();
    });
  });

  /* ---------------- Reset JSON ---------------- */
  resetJsonBtn.addEventListener("click", function () {
    formatEl.value = DEFAULT_JSON;
    formatEl.classList.remove("invalid");
    updatePreview();
  });

  /* ---------------- Live API preview ---------------- */
  function parseResponseFormat() {
    // Throws a SyntaxError with a readable message when JSON is malformed.
    var raw = formatEl.value.trim();
    if (!raw) {
      throw new Error("JSON пуст");
    }
    var parsed = JSON.parse(raw);
    return parsed;
  }

  function updatePreview() {
    var stop = stopEl.value.trim();
    var apiConfig = {
      response_format: null,
      max_tokens: null,
      stop: stop ? [stop] : null
    };

    try {
      apiConfig.response_format = parseResponseFormat();
      formatEl.classList.remove("invalid");
      // JSON must be an object (basic shape hint; backend validates fully)
      if (typeof apiConfig.response_format !== "object" || apiConfig.response_format === null) {
        throw new Error("JSON должен быть объектом");
      }
      previewEl.classList.remove("preview-invalid");
    } catch (err) {
      formatEl.classList.add("invalid");
      previewEl.classList.add("preview-invalid");
      previewEl.textContent = "Некорректный JSON: " + err.message;
      return;
    }

    var mt = parseInt(maxTokensEl.value, 10);
    apiConfig.max_tokens = Number.isInteger(mt) ? mt : null;

    previewEl.textContent = JSON.stringify(apiConfig, null, 2);
  }

  messageEl.addEventListener("input", updatePreview);
  formatEl.addEventListener("input", updatePreview);
  maxTokensEl.addEventListener("input", updatePreview);
  stopEl.addEventListener("input", updatePreview);
  updatePreview();

  /* ---------------- Submission ---------------- */
  form.addEventListener("submit", function (event) {
    event.preventDefault();
    submitComparison();
  });

  function submitComparison() {
    if (inFlight) return; // prevent duplicate submissions

    var message = messageEl.value.trim();
    if (!message) {
      showError("Введите запрос — он будет отправлен дважды.");
      return;
    }

    var responseFormat;
    try {
      responseFormat = parseResponseFormat();
    } catch (err) {
      formatEl.classList.add("invalid");
      showError("Некорректный JSON: проверьте синтаксис. (" + err.message + ")");
      formatEl.focus();
      return; // do NOT send, do NOT spend API balance
    }

    var maxTokens = parseInt(maxTokensEl.value, 10);
    if (!Number.isInteger(maxTokens) || maxTokens < 16 || maxTokens > 2000) {
      showError("Максимальная длина ответа: целое число от 16 до 2000 токенов.");
      return;
    }

    var payload = {
      message: message,
      response_format: responseFormat,
      max_tokens: maxTokens
    };
    var stopRaw = stopEl.value.trim();
    if (stopRaw) {
      payload.stop_sequence = stopRaw;
    }

    setLoading(true);
    hideError();
    hideResults();

    fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, status: response.status, data: data };
        });
      })
      .then(function (res) {
        if (!res.ok) {
          throw new Error(res.data.detail || ("Ошибка запроса (" + res.status + ")"));
        }
        render(res.data);
      })
      .catch(function (err) {
        showError(err.message || "Что-то пошло не так. Попробуйте ещё раз.");
      })
      .finally(function () {
        setLoading(false);
      });
  }

  /* ---------------- Rendering ---------------- */
  function render(data) {
    // Unrestricted card — raw provider text, actual finish_reason.
    answerUnrestricted.textContent = data.unrestricted.answer || "";
    finishUnrestricted.textContent = data.unrestricted.finish_reason || "—";

    // Controlled card
    if (data.controlled) {
      // Display provider content verbatim; pretty-print it ONLY if it is
      // valid JSON (json_object mode) — never rewrite or fabricate.
      answerControlled.textContent = displayAnswer(
        data.controlled.answer || "",
        data.controlled.settings
      );
      finishControlled.textContent = data.controlled.finish_reason || "—";
      appliedHeading.textContent = "Применённые API-параметры";
      appliedSettings.textContent = JSON.stringify(
        {
          response_format: data.controlled.settings && data.controlled.settings.response_format,
          max_tokens: data.controlled.settings && data.controlled.settings.max_tokens,
          stop: data.controlled.settings && data.controlled.settings.stop
        },
        null,
        2
      );
      controlledError.classList.add("hidden");
      controlledError.textContent = "";
    } else {
      // Controlled provider call failed. The REQUESTED parameters are known
      // before the call and are still displayed — labelled as requested,
      // not as applied. finish_reason is genuinely unavailable.
      answerControlled.textContent = "";
      finishControlled.textContent = "недоступна — запрос завершился ошибкой";
      appliedHeading.textContent = "Запрошенные API-параметры";
      appliedSettings.textContent = JSON.stringify(
        {
          response_format: data.settings && data.settings.response_format,
          max_tokens: data.settings && data.settings.max_tokens,
          stop: data.settings && data.settings.stop
        },
        null,
        2
      );
      controlledError.textContent =
        "Контролируемый запрос не выполнен: " +
        (data.controlled_error || "неизвестная ошибка.");
      controlledError.classList.remove("hidden");
    }

    renderSummary(data);
    resultsSection.classList.remove("hidden");
    summarySection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function displayAnswer(raw, settings) {
    // In json_object mode, show valid JSON nicely formatted for reading;
    // the raw model text is preserved for invalid JSON or text mode.
    var isJsonMode = settings && settings.response_format &&
      settings.response_format.type === "json_object";
    if (isJsonMode && raw) {
      try {
        return JSON.stringify(JSON.parse(raw), null, 2);
      } catch (err) {
        return raw; // provider violated json mode — show content as-is
      }
    }
    return raw;
  }

  function renderSummary(data) {
    var controlled = data.controlled;
    var finishC = controlled
      ? (controlled.finish_reason || "—")
      : "недоступна (запрос завершился ошибкой)";
    var items = [
      "Запрос отправлен в DeepSeek дважды — один и тот же текст, без изменений.",
      "Формат: response_format = " + JSON.stringify(
        data.settings ? data.settings.response_format : null
      ) + " — передан в API только в контролируемом запросе.",
      "Длина: max_tokens = " + (data.settings ? data.settings.max_tokens : "—") +
        " — лимит выходных токенов контролируемого запроса.",
      "Завершение: stop = " + JSON.stringify(
        data.settings ? data.settings.stop : null
      ) + " — стоп-последовательность контролируемого запроса.",
      "Причина завершения: без ограничений — " + (data.unrestricted.finish_reason || "—") +
        "; с ограничениями — " + finishC + ".",
      controlled ? "" : "Контролируемый запрос не выполнен — параметры показаны как запрошенные."
    ].filter(function (item) { return item !== ""; });
    summaryList.innerHTML = items
      .map(function (item) { return "<li>" + escapeHtml(item) + "</li>"; })
      .join("");
  }

  /* ---------------- Helpers ---------------- */
  function setLoading(on) {
    inFlight = on;
    button.disabled = on;
    button.textContent = on ? "Выполняется…" : "Сравнить ответы";
    loading.classList.toggle("hidden", !on);
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  }

  function hideError() {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
  }

  function hideResults() {
    resultsSection.classList.add("hidden");
    summarySection.classList.add("hidden");
  }

  function escapeHtml(text) {
    var div = document.createElement("div");
    div.textContent = String(text);
    return div.innerHTML;
  }
  }

  // The script sits at the end of <body>, but run safely even if it is ever
  // moved into <head>: wait for the DOM before wiring the controls.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initApp);
  } else {
    initApp();
  }
})();
