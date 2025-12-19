/* Bayleaf EPUB reader glue code.
   Expects epub.min.js to be loaded before this script.

   Required HTML elements (recommended IDs):
   - #viewer : container div for the EPUB rendition
   Optional controls:
   - [data-action="prev"] or #prev
   - [data-action="next"] or #next
   - #readingStatus (text status)

   The book URL is resolved in this order:
   1) <body data-book-url="...">
   2) window.BAYLEAF_BOOK_URL
   3) query params: ?book=... or ?file=... or ?path=...
*/

(() => {
  'use strict';

  const qs = (sel) => document.querySelector(sel);

  function getParam(name) {
    const u = new URL(window.location.href);
    return u.searchParams.get(name);
  }

  function setStatus(msg) {
    const el = qs('#readingStatus');
    if (el) el.textContent = msg;
  }

  function showFallback(msg) {
    const fallback = qs('#epub-fallback');
    if (fallback) fallback.style.display = 'block';
    if (msg) setStatus(msg);
  }

  function stableKey(input) {
    // Small stable hash for localStorage keys.
    // Not cryptographic. Just avoids huge keys.
    let h1 = 0x811c9dc5;
    for (let i = 0; i < input.length; i++) {
      h1 ^= input.charCodeAt(i);
      h1 = (h1 * 0x01000193) >>> 0;
    }
    return h1.toString(16).padStart(8, '0');
  }

  const SETTINGS_KEY = 'bayleaf:reader:settings';
  const DEFAULT_SETTINGS = {
    fontSize: 18,
    zoom: 1,
    background: 'parchment',
    text: 'ink',
  };

  const COLOR_PRESETS = {
    background: {
      parchment: '#f8f2e7',
      sepia: '#f3e3cd',
      mist: '#eff2f4',
      night: '#101214',
    },
    text: {
      ink: '#2b2722',
      coal: '#1b1b1b',
      slate: '#3a3f46',
      cream: '#f2ebdd',
    },
  };

  function loadSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (!raw) return { ...DEFAULT_SETTINGS };
      const parsed = JSON.parse(raw);
      return {
        ...DEFAULT_SETTINGS,
        ...parsed,
      };
    } catch {
      return { ...DEFAULT_SETTINGS };
    }
  }

  function saveSettings(settings) {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch {
      // ignore storage errors
    }
  }

  function normalizeSettings(settings) {
    const fontSize = Number(settings.fontSize) || DEFAULT_SETTINGS.fontSize;
    const zoom = Number(settings.zoom) || DEFAULT_SETTINGS.zoom;
    return {
      fontSize: Math.min(28, Math.max(12, fontSize)),
      zoom: Math.min(1.5, Math.max(0.7, zoom)),
      background: settings.background || DEFAULT_SETTINGS.background,
      text: settings.text || DEFAULT_SETTINGS.text,
    };
  }

  function applySettings(rendition, settings, viewer) {
    const normalized = normalizeSettings(settings);
    const bg = COLOR_PRESETS.background[normalized.background] || COLOR_PRESETS.background.parchment;
    const fg = COLOR_PRESETS.text[normalized.text] || COLOR_PRESETS.text.ink;
    const effectiveSize = Math.round(normalized.fontSize * normalized.zoom);

    if (viewer) {
      viewer.style.background = bg;
    }

    rendition.themes.override('body', 'background', `${bg} !important`);
    rendition.themes.override('body', 'color', `${fg} !important`);
    rendition.themes.override('a', 'color', `${fg} !important`);
    rendition.themes.override('html', 'color', `${fg} !important`);
    rendition.themes.override('body', 'font-size', `${effectiveSize}px !important`);
    rendition.themes.fontSize(`${effectiveSize}px`);

    const textSelectors = ['p', 'li', 'span', 'div', 'blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'];
    textSelectors.forEach((sel) => {
      rendition.themes.override(sel, 'color', `${fg} !important`);
    });
  }

  function bindSettingsUI(rendition, settings, viewer) {
    const fontSizeInput = qs('#settingFontSize');
    const zoomInput = qs('#settingZoom');
    const bgSelect = qs('#settingBackground');
    const textSelect = qs('#settingText');
    const fontSizeValue = qs('#settingFontSizeValue');
    const zoomValue = qs('#settingZoomValue');

    if (!fontSizeInput || !zoomInput || !bgSelect || !textSelect) return;

    function updateValueLabels(next) {
      if (fontSizeValue) fontSizeValue.textContent = `${next.fontSize}px`;
      if (zoomValue) zoomValue.textContent = `${Math.round(next.zoom * 100)}%`;
    }

    function syncUI(next) {
      fontSizeInput.value = String(next.fontSize);
      zoomInput.value = String(next.zoom);
      bgSelect.value = next.background;
      textSelect.value = next.text;
      updateValueLabels(next);
    }

    function readFromUI(current) {
      return normalizeSettings({
        ...current,
        fontSize: parseFloat(fontSizeInput.value),
        zoom: parseFloat(zoomInput.value),
        background: bgSelect.value,
        text: textSelect.value,
      });
    }

    function applyFromUI() {
      const next = readFromUI(settings);
      settings = next;
      applySettings(rendition, settings, viewer);
      saveSettings(settings);
      updateValueLabels(settings);
    }

    syncUI(settings);

    fontSizeInput.addEventListener('input', applyFromUI);
    zoomInput.addEventListener('input', applyFromUI);
    bgSelect.addEventListener('change', applyFromUI);
    textSelect.addEventListener('change', applyFromUI);
  }

  function resolveBookUrl() {
    const body = document.body;
    const fromAttr = body?.dataset?.bookUrl;
    if (fromAttr) return fromAttr;

    // Allow templates to inject a global.
    // Example: <script>window.BAYLEAF_BOOK_URL = "...";</script>
    // eslint-disable-next-line no-undef
    if (typeof window.BAYLEAF_BOOK_URL === 'string' && window.BAYLEAF_BOOK_URL) {
      return window.BAYLEAF_BOOK_URL;
    }

    // Fallback to query params
    return getParam('book') || getParam('file') || getParam('path') || '';
  }

  function ensureViewer() {
    const viewer = qs('#viewer');
    if (!viewer) {
      throw new Error('Missing #viewer element in read.html');
    }
    return viewer;
  }

  function bindControls(rendition) {
    const prevBtn = qs('[data-action="prev"]') || qs('#prev');
    const nextBtn = qs('[data-action="next"]') || qs('#next');

    if (prevBtn) {
      prevBtn.addEventListener('click', (e) => {
        e.preventDefault();
        rendition.prev();
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener('click', (e) => {
        e.preventDefault();
        rendition.next();
      });
    }

    // Keyboard navigation
    window.addEventListener('keydown', (e) => {
      // Ignore when typing in inputs
      const tag = (document.activeElement?.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || document.activeElement?.isContentEditable) return;

      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        rendition.prev();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        rendition.next();
      }
    });
  }

  async function start() {
    const viewer = ensureViewer();

    if (typeof window.ePub !== 'function') {
      showFallback('epub.js not loaded. Make sure /static/epub.min.js is included before reader.js.');
      return;
    }

    if (!window.JSZip) {
      showFallback('JSZip not loaded. Make sure /static/jszip.min.js is included before epub.min.js.');
      return;
    }

    const bookUrl = resolveBookUrl();
    if (!bookUrl) {
      showFallback('No book specified. Provide data-book-url on <body> or a ?book= URL parameter.');
      return;
    }

    const storageKey = `bayleaf:reader:loc:${stableKey(bookUrl)}`;
    const savedCfi = localStorage.getItem(storageKey) || '';

    setStatus('Loading book…');

    // Create book and rendition
    const book = window.ePub(bookUrl);

    // Use 100% height. Your CSS should set #viewer height (e.g. calc(100vh - header)).
    const rendition = book.renderTo(viewer, {
      width: '100%',
      height: '100%',
      spread: 'auto',
      flow: 'paginated',
    });

    // Apply a warm, book-like theme for readability.
    rendition.themes.register('bayleaf', {
      'body': {
        'background': '#f8f2e7 !important',
        'color': '#2b2722 !important',
        'line-height': '1.6 !important',
        'font-family': '"Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", "Georgia", serif !important',
        'font-size': '18px !important',
        'padding': '22px 30px !important',
      },
      'p': {
        'margin': '0 0 1em 0 !important',
      },
      'img': {
        'max-width': '100% !important',
        'height': 'auto !important',
      },
      'a': {
        'color': '#2b2722 !important',
        'text-decoration': 'underline',
      },
    });
    rendition.themes.select('bayleaf');
    const userSettings = loadSettings();
    applySettings(rendition, userSettings, viewer);
    bindSettingsUI(rendition, userSettings, viewer);

    // Persist location
    rendition.on('relocated', (location) => {
      try {
        const cfi = location?.start?.cfi;
        if (cfi) localStorage.setItem(storageKey, cfi);
      } catch {
        // ignore storage errors
      }
    });

    // Show a nicer title if the book metadata is available
    try {
      const metadata = await book.loaded.metadata;
      const title = metadata?.title;
      if (title) {
        const titleEl = qs('#bookTitle');
        if (titleEl) titleEl.textContent = title;
        document.title = `${title} · Bayleaf`;
      }
    } catch {
      // ignore
    }

    bindControls(rendition);

    try {
      if (savedCfi) {
        await rendition.display(savedCfi);
      } else {
        await rendition.display();
      }
      setStatus('');
    } catch (err) {
      console.error(err);
      showFallback('Could not open this EPUB. Check the URL and that the file is accessible from the container.');
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
