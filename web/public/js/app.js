(() => {
  const panel = document.querySelector('#assistant-panel');
  const stateLabel = document.querySelector('#state-label');
  const detail = document.querySelector('#assistant-detail');
  const transcriptLine = document.querySelector('#transcript-line');
  const toast = document.querySelector('#toast');
  const video = document.querySelector('#camera-video');
  const cameraView = document.querySelector('#camera-view');
  const overlay = document.querySelector('#camera-overlay');
  const placeholder = document.querySelector('#camera-placeholder');
  const cameraStatus = document.querySelector('#camera-status');
  const cameraFootnote = document.querySelector('#camera-footnote');
  const previewBadge = document.querySelector('#preview-badge');
  const cameraPanel = document.querySelector('#camera-panel');
  const startButton = document.querySelector('#start-listening');
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const backend = new window.BYAKUConnection();
  let recognition = null;
  let frameProcessor = null;
  let currentAudio = null;
  let initialSceneSpoken = false;
  let toastTimer = null;
  let state = 'idle';

  function setState(next, message = '') {
    state = next;
    panel.dataset.state = next;
    const content = {
      idle: ['AI Assistant', 'Say “START” to begin', 'Idle'],
      listening: ['Listening...', 'Say “START” or ask a question', 'Listening'],
      processing: ['Understanding your surroundings...', 'Starting local camera preview', 'Analyzing'],
      speaking: ['BYAKU is speaking', 'AI analysis is not connected yet', 'Speaking'],
    }[next];
    detail.textContent = message || content[1];
    stateLabel.textContent = content[2];
  }

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add('is-visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('is-visible'), 4200);
  }

  async function enablePreview() {
    try {
      await window.BYAKUCamera.start(
        video, cameraView, placeholder, cameraStatus, cameraFootnote, previewBadge, null,
      );
      return true;
    } catch (error) {
      showToast(error.message || 'Unable to access the camera. Check browser permissions.');
      cameraFootnote.textContent = 'Camera permission unavailable';
      return false;
    }
  }

  function playNarration(text, audioData) {
    const finish = () => {
      currentAudio = null;
      setState('idle', 'Camera analysis is active. Ask a question.');
      window.setTimeout(startListening, 350);
    };
    setState('speaking', text);
    window.speechSynthesis?.cancel();
    currentAudio?.pause();

    if (audioData) {
      currentAudio = new Audio(`data:audio/wav;base64,${audioData}`);
      currentAudio.onended = finish;
      currentAudio.onerror = () => speakWithBrowser(text, finish);
      currentAudio.play().catch(() => speakWithBrowser(text, finish));
      return;
    }
    speakWithBrowser(text, finish);
  }

  function speakWithBrowser(text, finish) {
    currentAudio = null;
    if (!('speechSynthesis' in window) || !text) {
      finish();
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onend = finish;
    utterance.onerror = finish;
    window.speechSynthesis.speak(utterance);
  }

  function handleBackendMessage(message) {
    if (message.type === 'error') {
      showToast(`${message.stage || 'Backend'}: ${message.message || 'Request failed'}`);
      setState('idle', 'Backend error. Camera preview remains available.');
      return;
    }
    if (message.type === 'status') {
      const cameraState = message.stage_status?.camera;
      if (cameraState === 'connected') {
        cameraStatus.innerHTML = '<span class="status-dot" aria-hidden="true"></span>AI processing';
      }
      return;
    }
    if (message.type === 'query_result') {
      const narration = message.narration || 'The backend returned no narration.';
      playNarration(narration, message.audio);
      return;
    }
    if (message.type !== 'frame_result') return;

    window.BYAKUFrameProcessor.drawOverlay(overlay, message.frame_size, message.objects);
    const objects = message.objects || [];
    cameraStatus.innerHTML = `<span class="status-dot" aria-hidden="true"></span>${message.stage_status?.detection === 'running' ? 'AI processing' : 'Detection unavailable'}`;
    cameraFootnote.textContent = `${objects.length} tracked object${objects.length === 1 ? '' : 's'} · ${message.processing_ms ?? '—'} ms`;

    if (message.scene_narration && !initialSceneSpoken) {
      initialSceneSpoken = true;
      playNarration(message.scene_narration, message.scene_audio);
    } else if (message.alert_audio) {
      const alert = (message.events || []).find((event) => event.message_required);
      if (alert?.narration) playNarration(alert.narration, message.alert_audio);
    }
  }

  async function startCameraSession(resumeListening = true) {
    if (frameProcessor) return true;
    cameraPanel.hidden = false;
    cameraPanel.parentElement.classList.add('camera-active');
    setState('processing', 'Opening camera and connecting to the AI backend.');
    const started = await enablePreview();
    if (!started) {
      setState('idle', 'Camera permission is needed to start analysis.');
      return false;
    }
    try {
      await backend.connect();
      frameProcessor = new window.BYAKUFrameProcessor(backend, video);
      frameProcessor.addEventListener('result', (event) => handleBackendMessage(event.detail));
      frameProcessor.start();
      setState('processing', 'Live camera frames are being analyzed.');
      cameraStatus.innerHTML = '<span class="status-dot" aria-hidden="true"></span>Connecting to Python';
      if (resumeListening) window.setTimeout(startListening, 500);
      return true;
    } catch (error) {
      setState('idle', 'Camera is on, but the Python backend is unavailable.');
      showToast(error.message || 'Could not connect to the Python backend.');
      return false;
    }
  }

  async function onTranscript(transcript) {
    const normalized = transcript.trim();
    transcriptLine.textContent = `Heard: “${normalized}”`;
    const isStart = /^start[.!?\s]*$/i.test(normalized);
    if (isStart) {
      await startCameraSession();
      return;
    }

    if (!frameProcessor && !(await startCameraSession(false))) return;
    setState('processing', 'Sending your question to the scene reasoning module.');
    if (!frameProcessor?.sendQuery(normalized)) {
      setState('idle', 'The backend connection is unavailable.');
      showToast('Could not send the query to the Python backend.');
    }
  }

  function startListening() {
    if (!Recognition) {
      showToast('Voice recognition is not supported here. Try a recent Chrome or Edge browser.');
      return;
    }
    if (recognition) {
      try { recognition.stop(); } catch { /* Already stopped. */ }
    }
    recognition = new Recognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => {
      setState('listening');
      transcriptLine.textContent = '';
    };
    recognition.onresult = (event) => onTranscript(event.results[0][0].transcript);
    recognition.onerror = (event) => {
      setState('idle');
      const message = event.error === 'not-allowed'
        ? 'Microphone permission was denied.'
        : 'No clear speech was received. Please try again.';
      showToast(message);
    };
    recognition.onend = () => {
      if (state === 'listening') setState('idle');
    };
    try {
      recognition.start();
    } catch {
      setState('idle');
      showToast('Voice recognition could not start. Please try again.');
    }
  }

  startButton.addEventListener('click', startListening);
  window.addEventListener('pagehide', () => {
    frameProcessor?.stop();
    backend.close();
    if (window.BYAKUCamera.stream) {
      window.BYAKUCamera.stop(video, cameraView, placeholder, cameraStatus, cameraFootnote, previewBadge, null);
    }
    window.speechSynthesis?.cancel();
  });
})();
