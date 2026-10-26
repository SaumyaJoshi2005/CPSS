"use strict";

const el = (id) => document.getElementById(id);
let info;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

async function request(url, body) {
  const options = body
    ? {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    : {};
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) {
    const error = new Error(value.error || "Request failed");
    error.status = response.status;
    error.loading = value.status === "loading";
    throw error;
  }
  return value;
}

function updateAvailability() {
  if (!info) return;
  const available = info.views[el("view").value]?.dense_available;
  for (const option of el("method").options) {
    option.disabled = !available && option.value !== "bm25";
  }
  if (!available) el("method").value = "bm25";
}

function renderCoverage(value) {
  info = value;
  const features = value.features || { review: true, classifier_boost: true };
  document.querySelectorAll('[data-feature="review"]').forEach((link) => {
    link.hidden = !features.review;
  });
  el("classifier-option").hidden = !features.classifier_boost;
  el("classifier-boost").disabled = !features.classifier_boost;
  if (!features.classifier_boost) el("classifier-boost").checked = false;
  el("coverage").replaceChildren(
    node("strong", `${value.searchable_count.toLocaleString()} searchable problems`),
    node("div", `of ${value.complete_corpus_count.toLocaleString()} complete corpus problems`),
    node("div", "Preserved training split"),
  );
  for (const option of el("view").options) {
    option.disabled = !value.views[option.value];
  }
  if (!value.views[el("view").value]) el("view").value = Object.keys(value.views)[0];
  updateAvailability();
}

function readSearchOptions() {
  const rating = (id) => (el(id).value === "" ? null : Number(el(id).value));
  return {
    query: el("query").value,
    method: el("method").value,
    view: el("view").value,
    rating_min: rating("rating-min"),
    rating_max: rating("rating-max"),
    tag_boost: el("tag-boost").checked,
    classifier_boost: el("classifier-boost").checked,
  };
}

function renderInterpretation(context) {
  let text = context.explicit_algorithms.length
    ? "Explicit techniques: " + context.explicit_algorithms.map((a) => a.named_text).join(", ")
    : "Using your description as written.";
  if (context.rating_min !== null || context.rating_max !== null) {
    text += ` · Rating ${context.rating_min ?? "any"}–${context.rating_max ?? "any"}`;
  }
  if (context.excluded_tags.length) {
    text += " · Excluding original tags: " + context.excluded_tags.join(", ");
  }
  el("interpretation").textContent = text;
}

function renderScoreDetails(hit, semantics) {
  const detail = node("details", undefined, "score-detail");
  const classifier = hit.classifier_boost_applied
    ? "Classifier boost on; uncalibrated classifier signal: " +
      hit.classifier_signal.toFixed(3) + " (separate from original tags)."
    : "Classifier boost off.";
  const explanation = `${semantics}. Base score: ${hit.base_retrieval_score.toFixed(5)}; ` +
    `final score: ${hit.score.toFixed(5)}. ` +
    `Explicit-tag boost ${hit.explicit_tag_boost_applied ? "on" : "off"}. ${classifier}`;
  detail.append(node("summary", "Ranking details"), node("p", explanation));
  return detail;
}

function highlightedQuote(evidence) {
  const quote = node("blockquote", undefined, "quote");
  if (!evidence.matched_terms.length) {
    quote.textContent = evidence.quote;
    return quote;
  }
  const escaped = evidence.matched_terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp("\\b(" + escaped.join("|") + ")\\b", "gi");
  let offset = 0;
  for (const match of evidence.quote.matchAll(pattern)) {
    quote.append(node("span", evidence.quote.slice(offset, match.index)));
    quote.append(node("mark", match[0]));
    offset = match.index + match[0].length;
  }
  quote.append(node("span", evidence.quote.slice(offset)));
  return quote;
}

function renderResult(hit, semantics) {
  const card = node("article", undefined, "result");
  const head = node("div", undefined, "result-head");
  head.append(node("span", String(hit.rank).padStart(2, "0"), "rank"));

  const title = node("div");
  const heading = node("h2");
  const link = node("a", hit.title);
  link.href = hit.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  heading.append(link);
  title.append(heading);

  const metadata = node("div", hit.problem_id, "meta");
  metadata.append(node("span", `Rating ${hit.rating}`, "rating"));
  title.append(metadata);
  head.append(title);
  card.append(head);

  const tags = node("div", undefined, "tags");
  tags.setAttribute("aria-label", "Original upstream tags");
  for (const tag of hit.original_tags) {
    const matching = hit.evidence.original_tag_overlap.includes(tag);
    tags.append(node("span", tag, "tag" + (matching ? " match" : "")));
  }
  card.append(tags);
  const evidence = node("div", undefined, "evidence-panel");
  const sourceLabel = hit.evidence.source_kind === "editorial" ? "Editorial excerpt" : "Statement excerpt";
  evidence.append(node("div", sourceLabel, "evidence-label"), highlightedQuote(hit.evidence));
  card.append(evidence);
  const matchingTerms = hit.evidence.matched_terms.length
    ? "Matched terms: " + hit.evidence.matched_terms.join(", ")
    : "Semantic candidate; no lexical overlap in this excerpt.";
  const footer = node("div", undefined, "result-footer");
  const sourceLink = node("a", "Open on Codeforces ↗", "source-link");
  sourceLink.href = hit.url;
  sourceLink.target = "_blank";
  sourceLink.rel = "noopener noreferrer";
  footer.append(node("div", matchingTerms, "matched"), sourceLink, renderScoreDetails(hit, semantics));
  card.append(footer);
  return card;
}

function renderResults(response) {
  el("results").replaceChildren();
  el("status").replaceChildren(
    node("strong", `${response.results.length} results · ${response.eligible_count.toLocaleString()} eligible problems`),
    node("span", `${(response.query_seconds * 1000).toFixed(0)} ms`, "time-pill"),
  );
  renderInterpretation(response.query_context);
  for (const hit of response.results) {
    el("results").append(renderResult(hit, response.score_semantics));
  }
  if (!response.results.length) {
    el("results").append(node("div", "No matching problems. Try a wider rating range or another description.", "empty"));
  }
}

function setBusy(busy) {
  el("search-button").disabled = busy;
  el("reset-search").disabled = busy;
  el("results").setAttribute("aria-busy", String(busy));
  document.querySelectorAll(".example").forEach((button) => { button.disabled = busy; });
}

function resetSearch() {
  el("query").value = "";
  el("rating-min").value = "";
  el("rating-max").value = "";
  el("method").value = "hybrid";
  el("view").value = info?.views.editorial ? "editorial" : Object.keys(info?.views || {})[0] || "editorial";
  el("tag-boost").checked = false;
  el("classifier-boost").checked = false;
  el("results").replaceChildren();
  el("interpretation").textContent = "";
  el("status").textContent = "Ready when you are";
  el("welcome").hidden = false;
  updateAvailability();
  el("query").focus?.();
}

async function handleSearch(event) {
  event.preventDefault();
  setBusy(true);
  el("welcome").hidden = true;
  el("status").textContent = "Searching your problem collection…";
  el("results").replaceChildren(node("div", "Finding problems and their source evidence…", "loading-state"));
  el("interpretation").textContent = "";
  try {
    renderResults(await request("/api/search", readSearchOptions()));
  } catch (error) {
    el("status").textContent = "Search needs your attention";
    const message = node("div", error.message, "error");
    message.setAttribute("role", "alert");
    el("results").replaceChildren(message);
  } finally {
    setBusy(false);
  }
}

async function initializeApplication() {
  setBusy(true);
  const deadline = Date.now() + 125000;
  while (true) {
    try {
      renderCoverage(await request("/api/info"));
      setBusy(false);
      el("status").textContent = "Ready when you are";
      return;
    } catch (error) {
      if (error.status === 503 && error.loading && Date.now() < deadline) {
        el("status").textContent = "Loading the verified search collection…";
        await new Promise((resolve) => setTimeout(resolve, 2000));
      } else {
        el("coverage").textContent = "Search is temporarily unavailable.";
        el("status").textContent = "Search is unavailable. Refresh the page to retry.";
        return;
      }
    }
  }
}
el("view").addEventListener("change", updateAvailability);
el("search-form").addEventListener("submit", handleSearch);
el("reset-search").addEventListener("click", resetSearch);
document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => {
    el("query").value = button.dataset.query;
    el("search-form").requestSubmit();
  });
});
initializeApplication();
