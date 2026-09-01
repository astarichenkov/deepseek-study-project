/* DeepSeek Study App frontend logic.
 * Vanilla JavaScript: no framework. Calls the REST API with fetch().
 */
(function () {
  "use strict";

  const form = document.getElementById("chat-form");
  const input = document.getElementById("message");
  const button = document.getElementById("send-button");
  const loading = document.getElementById("loading");
  const result = document.getElementById("result");
  const answer = document.getElementById("answer");
  const errorBox = document.getElementById("error");

  let inFlight = false;

  async function submitMessage() {
    if (inFlight) return; // prevent accidental repeated submission

    const message = input.value.trim();
    if (!message) {
      showError("Please enter a question first.");
      return;
    }

    setLoading(true);
    hideError();
    result.classList.add("hidden");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message }),
      });

      let data = {};
      try {
        data = await response.json();
      } catch (err) {
        // Non-JSON error body; fall through with empty data.
      }

      if (!response.ok) {
        throw new Error(
          data.detail || "Request failed. Please try again."
        );
      }

      answer.textContent = data.answer || "";
      result.classList.remove("hidden");
    } catch (err) {
      showError(err.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function setLoading(on) {
    inFlight = on;
    button.disabled = on;
    button.textContent = on ? "Sending…" : "Send";
    loading.classList.toggle("hidden", !on);
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  }

  function hideError() {
    errorBox.classList.add("hidden");
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    submitMessage();
  });

  // Ctrl+Enter (or Cmd+Enter on macOS) submits from anywhere in the textarea.
  input.addEventListener("keydown", function (event) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      submitMessage();
    }
  });
})();
