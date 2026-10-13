const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const welcomePage = fs.readFileSync(path.join(__dirname, '../public/welcome/index.html'), 'utf8');
const assistantPage = fs.readFileSync(path.join(__dirname, '../public/assistant.html'), 'utf8');

test('welcome Continue goes directly to the voice-start page', () => {
  assert.match(welcomePage, /class="skip-button" href="\/assistant\.html">Continue<\/a>/);
  assert.doesNotMatch(welcomePage, /Open the assistant|Explore project workflow/);
  assert.doesNotMatch(welcomePage, /index-DGY3_KB3\.js/);
  assert.match(assistantPage, /id="start-listening"/);
});

test('assistant camera preview preserves camera aspect ratio for video and overlays', () => {
  const assistantStyles = fs.readFileSync(path.join(__dirname, '../public/css/assistant-runtime.css'), 'utf8');
  assert.match(assistantStyles, /\.assistant-camera-view\.has-stream\s*\{[^}]*min-height:\s*0/);
  assert.match(assistantStyles, /\.assistant-camera-view\.has-stream\s*\{[^}]*width:\s*min\(100%,\s*640px\)/);
  assert.match(assistantStyles, /\.assistant-camera-view \.camera-feed\s*\{[^}]*aspect-ratio:\s*auto/);
  assert.doesNotMatch(assistantStyles, /\.assistant-camera-view \.camera-feed\s*\{[^}]*aspect-ratio:\s*16\s*\/\s*9/);
});
