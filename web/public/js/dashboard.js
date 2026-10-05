(() => {
  const video = document.querySelector('#dash-camera-video');
  const overlay = document.querySelector('#dash-camera-overlay');
  const container = document.querySelector('#dash-camera-view');
  const placeholder = document.querySelector('#dash-camera-placeholder');
  const cameraStatus = document.querySelector('#dash-camera-status');
  const detail = document.querySelector('#dash-camera-detail');
  const button = document.querySelector('#dash-camera-toggle');
  const connection = new window.BYAKUConnection();
  let frameProcessor = null;

  function statusText(value) {
    return String(value || 'unavailable').replaceAll('_', ' ');
  }

  function setText(selector, value) {
    const element = document.querySelector(selector);
    if (element) element.textContent = statusText(value);
  }

  function updatePipeline(status) {
    const keys = {
      camera: 'camera',
      detection: 'detection',
      tracking: 'tracking',
      depth: 'depth',
      spatial: 'spatial',
      scene: 'scene_graph',
      reasoning: 'reasoning',
      narration: 'narration',
      tts: 'tts',
    };
    document.querySelectorAll('.pipeline-track li[data-module]').forEach((item) => {
      const stage = keys[item.dataset.module];
      const value = status?.[stage] || 'unavailable';
      item.dataset.status = ['ready', 'connected'].includes(value) ? 'running' : value;
      const node = item.querySelector('.pipeline-node');
      let detailLabel = node.querySelector('[data-stage-status]');
      if (!detailLabel) {
        detailLabel = document.createElement('small');
        detailLabel.dataset.stageStatus = '';
        node.append(detailLabel);
      }
      detailLabel.textContent = statusText(value).toUpperCase();
      const stageTime = status?.stage_timings_ms?.[stage];
      node.title = `${statusText(value)}${Number.isFinite(stageTime) ? ` · ${stageTime.toFixed(1)} ms` : ''}`;
    });

    setText('#system-camera', status?.camera === 'connected' ? 'Connected' : 'Disconnected');
    setText('#system-detection', status?.detection);
    setText('#system-tracking', status?.tracking);
    setText('#system-depth', status?.depth);
    setText('#system-tts', status?.tts);
    setText('#system-fps', Number.isFinite(status?.processing_fps) ? `${status.processing_fps.toFixed(1)} FPS` : '—');
    setText('#system-latency', Number.isFinite(status?.processing_ms) ? `${status.processing_ms.toFixed(1)} ms` : '—');

    const connectionPill = document.querySelector('.dashboard-head-status .connection-pill span:last-child');
    if (connectionPill) connectionPill.textContent = status?.backend === 'connected' ? 'Python backend connected' : 'Python backend disconnected';
    const sidebarStatus = document.querySelector('.dashboard-sidebar-status strong');
    if (sidebarStatus) sidebarStatus.textContent = status?.backend === 'connected' ? 'Backend connected' : 'Backend disconnected';
  }

  function replaceRows(selector, rows, values, emptyText) {
    const body = document.querySelector(selector);
    if (!body) return;
    body.replaceChildren();
    if (!rows.length) {
      const row = document.createElement('tr');
      row.className = 'empty-row';
      const cell = document.createElement('td');
      cell.colSpan = values.length;
      cell.textContent = emptyText;
      row.append(cell);
      body.append(row);
      return;
    }
    for (const item of rows) {
      const row = document.createElement('tr');
      for (const value of values(item)) {
        const cell = document.createElement('td');
        cell.textContent = String(value ?? '—');
        row.append(cell);
      }
      body.append(row);
    }
  }

  function renderQuery(query, reasoning, narration) {
    const badge = document.querySelector('#query .data-tag');
    const strong = document.querySelector('.query-empty strong');
    const description = document.querySelector('.query-empty p');
    const object = reasoning?.objects?.[0];
    if (badge) badge.textContent = query ? 'Answered' : 'No query received';
    if (strong) strong.textContent = narration || 'Waiting for a question';
    if (description) description.textContent = query ? `“${query}”` : 'Query, target position, distance, and spoken answer will appear here.';
    setText('.query-result-list div:nth-child(1) dd', reasoning?.target || '—');
    setText('.query-result-list div:nth-child(2) dd', object?.horizontal_position || '—');
    setText('.query-result-list div:nth-child(3) dd', object?.distance_category || '—');
  }

  function renderFrame(message) {
    const objects = Array.isArray(message.objects) ? message.objects : [];
    const detections = Array.isArray(message.detections) ? message.detections : [];
    const events = Array.isArray(message.events) ? message.events : [];
    const tracks = Array.isArray(message.tracks) ? message.tracks : [];

    window.BYAKUFrameProcessor.drawOverlay(overlay, message.frame_size, objects);
    detail.textContent = `${tracks.length} tracked · ${message.processing_ms ?? '—'} ms`;
    cameraStatus.innerHTML = `<span class="status-dot" aria-hidden="true"></span>${message.stage_status?.camera === 'connected' ? 'AI processing' : 'Camera disconnected'}`;
    updatePipeline(message.stage_status || {});

    const count = document.querySelector('#detection-count');
    if (count) count.textContent = `${detections.length} detection${detections.length === 1 ? '' : 's'}`;
    replaceRows('#detection-rows', objects, (item) => [
      item.class_name,
      item.track_id,
      Number.isFinite(Number(item.confidence)) ? `${(Number(item.confidence) * 100).toFixed(1)}%` : '—',
      Array.isArray(item.bounding_box) ? item.bounding_box.map((value) => Math.round(value)).join(', ') : '—',
    ], 'No detections received');
    replaceRows('#spatial-rows', objects, (item) => [item.class_name, item.horizontal_position, item.distance_category], 'No spatial objects received');

    const preview = document.querySelector('#depth-preview');
    const depthStatus = document.querySelector('#depth-status');
    const depthMessage = document.querySelector('#depth-message');
    const surface = document.querySelector('.depth-surface');
    if (message.depth_preview && preview) {
      preview.src = `data:image/jpeg;base64,${message.depth_preview}`;
      preview.hidden = false;
      if (surface) surface.hidden = true;
      if (depthStatus) depthStatus.textContent = 'Live relative depth';
      if (depthMessage) depthMessage.textContent = 'Depth Anything V2 output';
    } else if (depthStatus) {
      depthStatus.textContent = message.stage_status?.depth === 'error' ? 'Depth unavailable' : 'No depth frame';
      if (depthMessage) depthMessage.textContent = message.stage_status?.depth === 'error' ? 'Depth processing unavailable' : 'Depth output will appear here';
    }

    const graph = document.querySelector('.graph-empty');
    const graphNode = graph?.querySelector('.graph-node');
    const graphText = graph?.querySelector('p');
    if (graphNode) graphNode.textContent = objects.length
      ? objects.map((item) => `${item.class_name} #${item.track_id}`).join(' · ')
      : 'No scene objects';
    if (graphText) graphText.textContent = message.relationships?.length
      ? message.relationships.map((edge) => `${edge.source} ${edge.relationship} ${edge.target}`).join(' · ')
      : 'No current relationships';
    const graphBadge = document.querySelector('#scene-graph .data-tag');
    if (graphBadge) graphBadge.textContent = `${objects.length} nodes · ${message.relationships?.length || 0} edges`;

    const eventList = document.querySelector('#event-list');
    const eventEmpty = document.querySelector('.events-empty');
    if (eventList) {
      eventList.replaceChildren();
      for (const event of events) {
        const item = document.createElement('li');
        item.textContent = `${event.event_type.replaceAll('_', ' ')} · ${event.class_name} #${event.track_id} · ${event.priority}`;
        eventList.append(item);
      }
    }
    if (eventEmpty) eventEmpty.hidden = events.length > 0;
    const eventBadge = document.querySelector('#events .data-tag');
    if (eventBadge) eventBadge.textContent = events.length ? `${events.length} updates` : 'No events';
  }

  async function refreshStatus() {
    try {
      const response = await fetch('/api/status', { cache: 'no-store' });
      if (!response.ok) throw new Error(`Backend status returned ${response.status}`);
      const status = await response.json();
      updatePipeline(status);
      if (status.last_frame) renderFrame(status.last_frame);
      if (status.last_query) renderQuery(status.last_query, status.last_reasoning, status.last_narration);
    } catch {
      updatePipeline({ backend: 'disconnected', camera: 'disconnected' });
    }
  }

  async function startCamera() {
    try {
      await window.BYAKUCamera.start(video, container, placeholder, cameraStatus, detail, null, button);
      await connection.connect();
      frameProcessor = new window.BYAKUFrameProcessor(connection, video);
      frameProcessor.addEventListener('result', (event) => {
        const message = event.detail;
        if (message.type === 'frame_result') renderFrame(message);
        else if (message.type === 'error') {
          detail.textContent = `${message.stage}: ${message.message}`;
          cameraStatus.innerHTML = '<span class="status-dot" aria-hidden="true"></span>Backend error';
        }
      });
      frameProcessor.start();
      button.textContent = 'Stop camera preview';
    } catch (error) {
      detail.textContent = error.message || 'Python backend unavailable';
      cameraStatus.innerHTML = '<span class="status-dot" aria-hidden="true"></span>Backend unavailable';
      if (window.BYAKUCamera.stream) {
        window.BYAKUCamera.stop(video, container, placeholder, cameraStatus, detail, null, button);
      }
    }
  }

  button.addEventListener('click', async () => {
    if (frameProcessor) {
      frameProcessor.stop();
      frameProcessor = null;
      connection.close();
      window.BYAKUCamera.stop(video, container, placeholder, cameraStatus, detail, null, button);
      updatePipeline({ backend: 'disconnected', camera: 'disconnected' });
      return;
    }
    await startCamera();
  });

  document.querySelectorAll('.nav-item').forEach((link) => {
    link.addEventListener('click', () => {
      document.querySelector('.nav-item.is-current')?.classList.remove('is-current');
      link.classList.add('is-current');
      document.querySelector('.dashboard-sidebar')?.classList.remove('is-menu-open');
      document.querySelector('#workflow-menu-toggle')?.setAttribute('aria-expanded', 'false');
    });
  });

  const menuToggle = document.querySelector('#workflow-menu-toggle');
  menuToggle?.addEventListener('click', () => {
    const expanded = menuToggle.getAttribute('aria-expanded') === 'true';
    menuToggle.setAttribute('aria-expanded', String(!expanded));
    menuToggle.setAttribute('aria-label', expanded ? 'Open project navigation' : 'Close project navigation');
    document.querySelector('.dashboard-sidebar')?.classList.toggle('is-menu-open', !expanded);
  });

  refreshStatus();
  window.setInterval(refreshStatus, 5000);
  window.addEventListener('pagehide', () => {
    frameProcessor?.stop();
    connection.close();
    if (window.BYAKUCamera.stream) {
      window.BYAKUCamera.stop(video, container, placeholder, cameraStatus, detail, null, button);
    }
  });
})();