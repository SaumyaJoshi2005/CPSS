"use strict";

const el = (id) => document.getElementById(id);
let pending = null;

function renderPendingReview(value) {
  el("progress").textContent = `${value.reviewed_pairs} of ${value.total_pairs} query–problem pairs reviewed`;
  pending = value.pending;
  if (!pending) {
    el("workspace").hidden = true;
    el("progress").textContent += " · Queue complete. Run retrieval-evaluate to compute the report.";
    return;
  }
  el("review-query").textContent = pending.query;
  el("review-title").textContent = pending.title;
  el("review-link").href = pending.url;
  el("review-statement").textContent = pending.statement;
  el("review-editorial").textContent = pending.editorial;
  el("review-source").textContent = `${pending.problem_id} · Rating ${pending.rating} · Open R1 · ` +
    "Frozen source versions retained with each judgment.";
  el("rationale").value = "";
  el("attestation").checked = false;
}

async function loadReview() {
  try {
    const response = await fetch("/api/review");
    const value = await response.json();
    if (!response.ok) throw new Error(value.error);
    renderPendingReview(value);
  } catch (error) {
    el("review-error").textContent = error.message;
    el("workspace").hidden = true;
  }
}

async function submitGrade(button) {
  if (!pending) return;
  const buttons = document.querySelectorAll("[data-grade]");
  buttons.forEach((item) => { item.disabled = true; });
  try {
    const response = await fetch("/api/judgments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pair_id: pending.pair_id,
        grade: Number(button.dataset.grade),
        rationale: el("rationale").value,
        reviewer: el("reviewer").value,
        human_attested: el("attestation").checked,
      }),
    });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error);
    el("review-error").textContent = "Decision saved. Next blinded pair:";
    await loadReview();
  } catch (error) {
    el("review-error").textContent = error.message;
  } finally {
    buttons.forEach((item) => { item.disabled = false; });
  }
}

document.querySelectorAll("[data-grade]").forEach((button) => {
  button.addEventListener("click", () => submitGrade(button));
});
loadReview();
