// Actual-local-server integration of the shipped client script with a minimal
// DOM adapter. This checks client behavior, not browser layout or rendering.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const base = process.argv[2] || 'http://127.0.0.1:8765';
const publicOrigin = process.argv[3];
class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.value = '';
    this.checked = false;
    this.disabled = false;
    this.events = {};
    this.dataset = {};
    this.attributes = {};
  }
  set textContent(value) {
    this.text = String(value);
    this.children = [];
  }
  get textContent() {
    return (this.text || '') + this.children.map(c => c.textContent || '').join('');
  }
  append(...values) {
    this.children.push(...values);
  }
  replaceChildren(...values) {
    this.text = '';
    this.children = [...values];
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }
  addEventListener(name, fn) {
    this.events[name] = fn;
  }
  focus() { this.focused = true; }
  requestSubmit() {
    this.lastSubmission = this.events.submit({ preventDefault() {} });
  }
}
const elements = new Map();
const get = id => {
  if (!elements.has(id))
    elements.set(id, new Element(id));
  return elements.get(id);
};
get('method').value = 'hybrid';
get('method').options = ['hybrid', 'bm25', 'dense'].map(value => ({
  value
}));
get('view').value = 'editorial';
get('view').options = ['editorial', 'statement'].map(value => ({
  value
}));
const exampleQueries = [
  'DSU with connectivity queries.',
  'Maintain connected groups as new connections are added.',
  'Binary search on the answer with a greedy feasibility check.',
];
const examples = exampleQueries.map(query => {
  const button = new Element('button');
  button.dataset.query = query;
  return button;
});
const document = {
  getElementById: get,
  createElement: tag => new Element(tag),
  querySelectorAll: selector => selector === '.example' ? examples
    : selector === '[data-feature="review"]' ? [get('review-link')] : [],
};
const context = vm.createContext({
  document, console, setTimeout,
  fetch: (url, options) => fetch(new URL(url, base), publicOrigin && options?.method === 'POST'
    ? { ...options, headers: { ...options.headers, Origin: publicOrigin } }
    : options)
});
vm.runInContext(fs.readFileSync('modeling/application/static/app.js', 'utf8'), context);
async function main() {
  for (let i = 0; i < 100 && !get('coverage').children.length; i++)
    await new Promise(resolve => setTimeout(resolve, 50));
  assert.match(get('coverage').textContent, /3,716 searchable problems/);
  assert.equal(get('review-link').hidden, Boolean(publicOrigin));
  assert.equal(get('classifier-option').hidden, Boolean(publicOrigin));
  assert.equal(get('classifier-boost').disabled, Boolean(publicOrigin));
  const queries = exampleQueries;
  const results = [];
  for (const [i, query] of queries.entries()) {
    get('method').value = ['bm25', 'dense', 'hybrid'][i];
    examples[i].events.click();
    assert.equal(get('query').value, query);
    assert.equal(get('search-button').disabled, true);
    assert.equal(get('results').attributes['aria-busy'], 'true');
    assert.equal(get('results').children[0].className, 'loading-state');
    assert.ok(examples.every(button => button.disabled));
    await get('search-form').lastSubmission;
    assert.equal(get('search-button').disabled, false);
    assert.equal(get('results').attributes['aria-busy'], 'false');
    assert.equal(get('welcome').hidden, true);
    assert.equal(get('results').children.length, 10);
    assert.ok(get('results').children.every(c => c.className === 'result'));
    assert.match(get('status').textContent, /10 results/);
    results.push({
      query, method: get('method').value, cards: 10
    });
  }
  const source = '<img src=x onerror=alert(1)> DSU and disjoint sets.';
  const excerpt = context.highlightedQuote({quote: source, matched_terms: ['dsu', 'onerror']});
  assert.equal(excerpt.textContent, source);
  assert.ok(excerpt.children.some(child => child.tag === 'mark'));
  assert.ok(excerpt.children.every(child => ['span', 'mark'].includes(child.tag)));
  get('rating-min').value = '4000';
  await get('search-form').events.submit({
    preventDefault() {
    }
  });
  assert.equal(get('results').children[0].className, 'empty');
  get('rating-min').value = '2200';
  get('rating-max').value = '1200';
  await get('search-form').events.submit({
    preventDefault() {
    }
  });
  assert.equal(get('results').children[0].className, 'error');
  assert.equal(get('results').children[0].attributes.role, 'alert');
  assert.match(get('results').textContent, /constraints conflict|Invalid search request/);
  get('rating-min').value = '';
  get('rating-max').value = '';
  get('query').value = '';
  await get('search-form').events.submit({
    preventDefault() {
    }
  });
  assert.match(get('results').textContent, /nonempty|Invalid search request/);
  get('tag-boost').checked = true;
  get('reset-search').events.click();
  assert.equal(get('results').children.length, 0);
  assert.equal(get('query').value, '');
  assert.equal(get('rating-min').value, '');
  assert.equal(get('rating-max').value, '');
  assert.equal(get('tag-boost').checked, false);
  assert.equal(get('welcome').hidden, false);
  assert.equal(get('query').focused, true);
  console.log(JSON.stringify({
    status: 'passed', scope: 'Actual HTTP API plus shipped client-script DOM behavior; no browser rendering claim',
    examples: results, empty_results: true, invalid_constraints: true, empty_input: true,
    loading_and_busy_state: true, reset_search: true, source_quote_highlighting_and_escaping: true
  }));
}
main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
