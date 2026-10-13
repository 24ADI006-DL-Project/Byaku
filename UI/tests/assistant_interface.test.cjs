const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const interfaceSource = fs.readFileSync(path.join(__dirname, '../public/js/assistant_interface.js'), 'utf8');

class FakeElement {
  constructor() {
    this.listeners = new Map();
    this.attributes = new Map();
    this.classList = {
      add() {},
      remove() {},
      toggle() {},
    };
    this.dataset = {};
    this.parentElement = { classList: this.classList };
    this.hidden = false;
    this.textContent = '';
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  setAttribute(name, value) {
    this.attributes.set(name, value);
  }
}

test('camera startup failure remains specific in the assistant status', async () => {
  const selectors = [
    '#assistant-panel',
    '#state-label',
    '#assistant-detail',
    '#transcript-line',
    '#toast',
    '#camera-video',
    '#camera-view',
    '#camera-overlay',
    '#camera-placeholder',
    '#camera-status',
    '#camera-footnote',
    '#preview-badge',
    '#camera-panel',
    '#start-listening',
    '#stop-assistant',
    '#text-query-form',
    '#text-query-input',
    '#assistant-action-label',
  ];
  const elements = new Map(selectors.map((selector) => [selector, new FakeElement()]));
  const window = {
    setTimeout,
    clearTimeout,
    addEventListener() {},
    BYAKUConnection: class extends EventTarget {},
    BYAKUCamera: {
      start: async () => { throw new Error('The camera is in use by another app or cannot be opened.'); },
      get stream() { return null; },
      stop() {},
    },
    BYAKUSpeechController: class { cancel() {} },
    BYAKULiveNarrationScheduler: class { stop() {} },
  };
  const context = vm.createContext({
    window,
    document: { querySelector: (selector) => elements.get(selector) || null },
    performance,
    setTimeout,
    clearTimeout,
  });
  vm.runInContext(interfaceSource, context);

  elements.get('#start-listening').listeners.get('click')();
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(elements.get('#camera-status').textContent, 'Camera unavailable');
  assert.equal(elements.get('#assistant-detail').textContent, 'The camera is in use by another app or cannot be opened.');
  assert.equal(elements.get('#toast').textContent, 'The camera is in use by another app or cannot be opened.');
});
