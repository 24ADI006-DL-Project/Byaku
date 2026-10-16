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
  let currentReviewFrame = null;
  let currentDetections = [];
  let selectedDetection = null;
  let reviewHistory = [];
  let reviewImageToken = 0;
  let reviewSessionId = window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  const reviewSubmissionIds = new Map();
  let selectedGraphNodeId = null;

  function statusText(value) {
    return String(value || 'unavailable').replaceAll('_', ' ');
  }

  function setText(selector, value) {
    const element = document.querySelector(selector);
    if (element) element.textContent = statusText(value);
  }

  function formatRelativeDepth(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : 'Unavailable';
  }

  function formatPosition(object) {
    return [object.horizontal_position, object.vertical_position]
      .filter((value) => typeof value === 'string' && value)
      .join(', ') || 'Unavailable';
  }

  function detectionKey(object) {
    if (object?.track_id !== null && object?.track_id !== undefined) return `track:${object.track_id}`;
    return `${object?.model_class_name || object?.class_name}:${(object?.bounding_box || []).join(',')}`;
  }

  function setReviewStatus(message, isError = false) {
    const status = document.querySelector('#review-feedback-status');
    if (!status) return;
    status.textContent = message;
    status.dataset.error = String(isError);
  }

  function drawReviewFrame(frameUrl, frameSize, objects) {
    const canvas = document.querySelector('#review-frame-canvas');
    const empty = document.querySelector('#review-frame-empty');
    if (!canvas || !empty || typeof frameUrl !== 'string') return;
    const token = ++reviewImageToken;
    const image = new Image();
    image.onload = () => {
      if (token !== reviewImageToken) return;
      const context = canvas.getContext('2d');
      if (!context) return;
      const width = Number(frameSize?.width) || image.naturalWidth;
      const height = Number(frameSize?.height) || image.naturalHeight;
      canvas.width = width;
      canvas.height = height;
      context.drawImage(image, 0, 0, width, height);
      const selectedKey = selectedDetection ? detectionKey(selectedDetection) : null;
      for (const object of objects) {
        const box = object.bounding_box;
        if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) continue;
        const [x1, y1, x2, y2] = box;
        context.lineWidth = Math.max(2, width / 320);
        context.strokeStyle = selectedKey === detectionKey(object) ? '#ffe26b' : '#56f0bc';
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const confidence = Number.isFinite(Number(object.confidence))
          ? ` ${(Number(object.confidence) * 100).toFixed(0)}%`
          : '';
        const track = object.track_id === null || object.track_id === undefined ? '' : ` #${object.track_id}`;
        const remembered = object.remembered_correction?.corrected_name;
        const predicted = object.model_class_name || object.class_name || 'object';
        const label = `${predicted}${remembered ? ` → ${remembered}` : ''}${track}${confidence}`;
        context.font = `600 ${Math.max(12, width / 48)}px sans-serif`;
        const labelWidth = context.measureText(label).width + 10;
        const labelTop = Math.max(0, y1 - 24);
        context.fillStyle = 'rgba(25, 52, 61, 0.88)';
        context.fillRect(x1, labelTop, labelWidth, 22);
        context.fillStyle = '#ffffff';
        context.fillText(label, x1 + 5, labelTop + 16);
      }
      empty.hidden = true;
      canvas.hidden = false;
    };
    image.onerror = () => {
      if (token !== reviewImageToken) return;
      canvas.hidden = true;
      empty.hidden = false;
      empty.textContent = 'The latest camera frame could not be displayed.';
    };
    image.src = frameUrl;
  }

  function renderReviewDetections() {
    const list = document.querySelector('#review-detection-list');
    const selectedLabel = document.querySelector('#review-selected');
    const confirmButton = document.querySelector('#review-confirm');
    const incorrectButton = document.querySelector('#review-incorrect');
    const correctionForm = document.querySelector('#review-correction-form');
    if (!list || !selectedLabel || !confirmButton || !incorrectButton || !correctionForm) return;
    const knownClasses = [...new Set(currentDetections
      .flatMap((object) => [object.model_class_name, object.class_name])
      .filter((name) => typeof name === 'string' && name.trim()))].sort();
    const classList = document.querySelector('#review-known-classes');
    if (classList) {
      classList.replaceChildren();
      for (const className of knownClasses) {
        const option = document.createElement('option');
        option.value = className;
        classList.append(option);
      }
    }
    list.replaceChildren();
    if (!currentDetections.length) {
      const emptyItem = document.createElement('li');
      emptyItem.className = 'review-empty';
      emptyItem.textContent = 'No current detections';
      list.append(emptyItem);
    } else {
      for (const object of currentDetections) {
        const item = document.createElement('li');
        const select = document.createElement('button');
        select.type = 'button';
        select.className = 'review-detection-option';
        select.classList.toggle('is-selected', selectedDetection && detectionKey(selectedDetection) === detectionKey(object));
        const track = object.track_id === null || object.track_id === undefined ? 'untracked' : `track #${object.track_id}`;
        const confidence = Number.isFinite(Number(object.confidence))
          ? `${(Number(object.confidence) * 100).toFixed(1)}%`
          : 'confidence unavailable';
        const remembered = object.remembered_correction?.corrected_name;
        const predicted = object.model_class_name || object.class_name || 'Unknown';
        select.textContent = `${predicted}${remembered ? ` → remembered: ${remembered}` : ''} · ${track} · ${confidence}`;
        select.addEventListener('click', () => {
          selectedDetection = object;
          renderReviewDetections();
          drawReviewFrame(currentReviewFrame?.dataUrl, currentReviewFrame?.size, currentDetections);
        });
        item.append(select);
        list.append(item);
      }
    }
    if (selectedDetection) {
      const track = selectedDetection.track_id === null || selectedDetection.track_id === undefined
        ? 'untracked'
        : `track #${selectedDetection.track_id}`;
      const predictedName = selectedDetection.model_class_name || selectedDetection.class_name || 'Unknown';
      selectedLabel.textContent = `Selected: ${predictedName}${selectedDetection.remembered_correction ? ` → remembered: ${selectedDetection.class_name}` : ''} · ${track}`;
      const predicted = document.querySelector('#review-predicted-name');
      if (predicted) predicted.textContent = predictedName;
    } else {
      selectedLabel.textContent = 'Select a detection from the list to review it.';
    }
    const hasSelection = Boolean(selectedDetection);
    const alreadyReviewed = hasSelection && reviewHistory.some((record) =>
      record.frame_id === String(currentReviewFrame?.id)
      && detectionKey(record.original_prediction) === detectionKey(selectedDetection)
      && record.review_status !== 'undone');
    confirmButton.disabled = !hasSelection || alreadyReviewed;
    incorrectButton.disabled = !hasSelection || alreadyReviewed;
    correctionForm.hidden = !hasSelection || correctionForm.dataset.open !== 'true';
    document.querySelector('#review-question').textContent = alreadyReviewed
      ? 'This detection has already been reviewed.'
      : 'Is this object detection correct?';
    if (alreadyReviewed) {
      const record = reviewHistory.find((item) =>
        item.frame_id === String(currentReviewFrame?.id)
        && detectionKey(item.original_prediction) === detectionKey(selectedDetection)
        && item.review_status !== 'undone');
      setReviewStatus(record?.human_corrected_name
        ? `Saved: model predicted “${selectedDetection.class_name}”; corrected to “${record.human_corrected_name}”.`
        : 'Confirmation saved for this detection.');
    }
  }

  function renderReviewHistory() {
    const list = document.querySelector('#review-history');
    if (!list) return;
    list.replaceChildren();
    if (!reviewHistory.length) {
      const empty = document.createElement('li');
      empty.textContent = 'No saved review actions.';
      list.append(empty);
      return;
    }
    for (const record of reviewHistory) {
      const item = document.createElement('li');
      const time = new Date(record.timestamp).toLocaleString();
      const label = record.original_prediction?.class_name || 'Unknown object';
      const corrected = record.human_corrected_name
        ? ` → ${record.human_corrected_name}`
        : record.review_status === 'confirmed' || record.review_status === 'accepted'
          ? ' · confirmed correct'
          : '';
      const source = record.source || record.frame_reference?.source || 'live';
      const sourceLabel = source === 'upload_video' || source === 'video'
        ? '[Video Upload]'
        : source === 'upload_image' || source === 'image'
          ? '[Image Upload]'
          : '[Live Camera]';
      const summary = document.createElement('span');
      summary.textContent = `${sourceLabel} ${time} · ${label}${record.track_id === null || record.track_id === undefined ? '' : ` #${record.track_id}`}${corrected} · ${record.review_status}`;
      item.append(summary);
      if (record.review_status !== 'undone') {
        const undo = document.createElement('button');
        undo.type = 'button';
        undo.className = 'text-button review-undo-item';
        undo.textContent = 'Undo';
        undo.addEventListener('click', () => undoReview(record.feedback_id));
        item.append(' ', undo);
      }
      list.append(item);
    }
    renderReviewDetections();
  }

  async function loadReviewHistory() {
    const response = await fetch('/api/feedback?limit=100', { cache: 'no-store' });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Feedback history returned ${response.status}`);
    reviewHistory = Array.isArray(body.feedback) ? body.feedback : [];
    renderReviewHistory();
  }
  if (typeof window !== 'undefined') {
    window.byakuReloadReviewHistory = loadReviewHistory;
  }

  function refreshReviewScene(message) {
    const objects = Array.isArray(message.objects) ? message.objects : [];
    const previousKey = selectedDetection ? detectionKey(selectedDetection) : null;
    const frameUrl = message.client_frame_data_url || message.frame_preview || message.last_frame_preview;
    const frameId = message.frame_id ?? message.frame_number ?? null;
    currentReviewFrame = {
      id: frameId,
      number: message.frame_number ?? null,
      size: message.frame_size,
      dataUrl: frameUrl || (currentReviewFrame?.id === frameId ? currentReviewFrame.dataUrl : null),
      reviewFrameToken: message.review_frame_token ?? null,
    };
    currentDetections = objects;
    if (previousKey) {
      selectedDetection = objects.find((object) => detectionKey(object) === previousKey) || null;
    }
    const frameLabel = document.querySelector('#review-frame-label');
    if (frameLabel) {
      const cameraConnected = message.stage_status?.camera === 'connected';
      frameLabel.textContent = currentReviewFrame.id === null
        ? 'Waiting for a live frame'
        : `${cameraConnected ? 'Live frame' : 'Last processed frame'} · ${cameraConnected ? 'camera active' : 'camera disconnected'} · ${objects.length} detections`;
    }
    const empty = document.querySelector('#review-frame-empty');
    if (!currentReviewFrame.dataUrl && empty) {
      empty.hidden = false;
      empty.textContent = 'A saved detection is available, but its image is not in this browser session. Start the dashboard camera preview to review a live frame.';
      document.querySelector('#review-frame-canvas').hidden = true;
    } else {
      drawReviewFrame(currentReviewFrame.dataUrl, currentReviewFrame.size, objects);
    }
    renderReviewDetections();
  }

  function stableSubmissionId(correctionType, correctedValue) {
    const key = JSON.stringify([
      currentReviewFrame?.id,
      detectionKey(selectedDetection),
      correctionType,
      correctedValue,
    ]);
    if (!reviewSubmissionIds.has(key)) {
      reviewSubmissionIds.set(key, window.crypto?.randomUUID?.() || `${reviewSessionId}-${reviewSubmissionIds.size}`);
    }
    return reviewSubmissionIds.get(key);
  }

  async function saveReview(correctionType, correctedValue, reviewStatus) {
    if (!selectedDetection || !currentReviewFrame) return;
    const reviewedDetection = {
      ...selectedDetection,
      class_name: selectedDetection.model_class_name || selectedDetection.class_name,
    };
    const controls = document.querySelectorAll('#review-confirm, #review-incorrect, #review-save');
    controls.forEach((control) => { control.disabled = true; });
    const payload = {
      session_id: reviewSessionId,
      frame_id: currentReviewFrame.id === null ? null : String(currentReviewFrame.id),
      frame_number: currentReviewFrame.number,
      review_frame_token: currentReviewFrame.reviewFrameToken,
      image_dimensions: currentReviewFrame.size || null,
      track_id: selectedDetection.track_id ?? null,
      original_prediction: reviewedDetection,
      corrected_value: correctedValue,
      human_corrected_name: correctionType === 'class_correction' ? correctedValue : null,
      correction_type: correctionType,
      review_status: reviewStatus,
      submission_id: stableSubmissionId(correctionType, correctedValue),
    };
    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `Feedback save returned ${response.status}`);
      document.querySelector('#review-correction-form').dataset.open = 'false';
      document.querySelector('#review-correction-form').reset();
      setReviewStatus(correctionType === 'class_correction'
        ? `Saved: model predicted “${reviewedDetection.class_name}”; corrected to “${correctedValue}”.`
        : 'Confirmation saved to persistent review history.');
      await loadReviewHistory();
    } catch (error) {
      setReviewStatus(`Could not save review: ${error.message}`, true);
    } finally {
      renderReviewDetections();
    }
  }

  async function submitCorrection(event) {
    event.preventDefault();
    const correctedName = document.querySelector('#review-corrected-label').value.trim();
    if (!correctedName) {
      setReviewStatus('A correct object name is required before saving.', true);
      return;
    }
    await saveReview('class_correction', correctedName, 'corrected');
  }

  async function undoReview(feedbackId) {
    try {
      const response = await fetch(`/api/feedback/${encodeURIComponent(feedbackId)}/undo`, { method: 'POST' });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `Undo returned ${response.status}`);
      setReviewStatus('Last saved review action was marked undone; history is preserved.');
      await loadReviewHistory();
    } catch (error) {
      setReviewStatus(`Could not undo review action: ${error.message}`, true);
    }
  }

  function updatePipeline(status) {
    const values = status?.stage_status
      ? { ...status, ...status.stage_status }
      : status || {};
    const keys = {
      camera: 'camera',
      detection: 'detection',
      tracking: 'tracking',
      depth: 'depth',
      spatial: 'spatial',
      scene: 'scene_graph',
      reasoning: 'reasoning',
      narration: 'narration',
      review: 'review',
    };
    document.querySelectorAll('.pipeline-track li[data-module]').forEach((item) => {
      const stage = keys[item.dataset.module];
      const value = stage === 'review'
        ? values.review || (values.backend === 'connected' ? 'available' : 'unavailable')
        : values[stage] || 'unavailable';
      item.dataset.status = ['ready', 'connected'].includes(value) ? 'running' : value;
      const node = item.querySelector('.pipeline-node');
      let detailLabel = node.querySelector('[data-stage-status]');
      if (!detailLabel) {
        detailLabel = document.createElement('small');
        detailLabel.dataset.stageStatus = '';
        node.append(detailLabel);
      }
      detailLabel.textContent = statusText(value).toUpperCase();
      const stageTime = values.stage_timings_ms?.[stage];
      node.title = `${statusText(value)}${Number.isFinite(stageTime) ? ` · ${stageTime.toFixed(1)} ms` : ''}`;
    });

    const cameraState = values.camera;
    setText('#system-camera', cameraState || 'unavailable');
    if (typeof values.last_frame_available === 'boolean') {
      const frameStatus = document.querySelector('#system-last-frame');
      if (frameStatus) {
        frameStatus.textContent = values.last_frame_available
          ? cameraState === 'connected' ? 'Available · live' : 'Available · retained'
          : 'None available';
      }
      if (!values.last_frame_available && cameraState !== 'connected') {
        detail.textContent = 'Camera is off · no processed frame is available';
        cameraStatus.innerHTML = '<span class="status-dot status-dot-muted" aria-hidden="true"></span>Camera disconnected · no retained frame';
      }
    }
    setText('#system-detection', values.detection);
    setText('#system-tracking', values.tracking);
    setText('#system-depth', values.depth);
    setText('#system-tts', values.tts);
    setText('#system-latency', Number.isFinite(values.processing_ms) ? `${values.processing_ms.toFixed(1)} ms` : '—');
    setText('#system-initialization', Number.isFinite(values.initialization_ms) ? `${values.initialization_ms.toFixed(1)} ms` : '—');
    setText('#system-timing-source', values.timing_source || 'not reported');

    const connectionPill = document.querySelector('.dashboard-head-status .connection-pill span:last-child');
    if (connectionPill) connectionPill.textContent = values.backend === 'connected' ? 'Python backend connected' : 'Python backend disconnected';
    const sidebarStatus = document.querySelector('.dashboard-sidebar-status strong');
    if (sidebarStatus) sidebarStatus.textContent = values.backend === 'connected' ? 'Backend connected' : 'Backend disconnected';
    const sidebarDetail = document.querySelector('.dashboard-sidebar-status small');
    if (sidebarDetail) {
      sidebarDetail.textContent = cameraState === 'connected'
        ? 'Camera connected'
        : values.last_frame_available ? 'Camera off · last frame retained' : 'Camera disconnected';
    }
    const overall = document.querySelector('.backend-state');
    if (overall) {
      overall.textContent = values.backend === 'connected'
        ? 'Live backend status'
        : 'Backend disconnected';
    }
    if (cameraState) {
      cameraStatus.innerHTML = cameraState === 'connected'
        ? '<span class="status-dot" aria-hidden="true"></span>Camera connected'
        : `<span class="status-dot status-dot-muted" aria-hidden="true"></span>${values.last_frame_available ? 'Camera disconnected · last frame retained' : 'Camera disconnected'}`;
    }
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
    if (description) description.textContent = query ? `“${query}”` : 'Query, target position, relative depth category, and spoken answer will appear here.';
    setText('.query-result-list div:nth-child(1) dd', object?.class_name || '—');
    setText('.query-result-list div:nth-child(2) dd', object ? formatPosition(object) : '—');
    setText('.query-result-list div:nth-child(3) dd', object ? formatRelativeDepth(object.relative_depth_value) : '—');
  }

  function renderNarration(text) {
    const narrationText = document.querySelector('#narration-text');
    const narrationStatus = document.querySelector('#narration-status');
    const available = typeof text === 'string' && text.trim().length > 0;
    if (narrationText) narrationText.textContent = available
      ? text.trim()
      : 'No narration text was returned for the latest processed frame.';
    if (narrationStatus) narrationStatus.textContent = available ? 'Text ready · playback unconfirmed' : 'No narration text';
  }

  function renderGraph(graph) {
    const svg = document.querySelector('#scene-graph-svg');
    const empty = document.querySelector('#scene-graph-empty');
    const inspector = document.querySelector('#graph-selection');
    const relationList = document.querySelector('#graph-relationships');
    const badge = document.querySelector('#scene-graph .data-tag');
    if (!svg || !empty || !inspector || !relationList || !badge) return;

    const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
    const edges = Array.isArray(graph?.edges) ? graph.edges : [];
    const nodeIds = new Set(nodes.map((node) => String(node.object_id ?? node.id ?? '')));
    const positionedNodes = new Map();
    const centerX = 300;
    const centerY = 104;
    const radius = Math.min(175, 42 + nodes.length * 12);
    nodes.forEach((node, index) => {
      const angle = nodes.length === 1 ? 0 : (Math.PI * 2 * index / nodes.length) - Math.PI / 2;
      positionedNodes.set(String(node.object_id ?? node.id ?? ''), {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * Math.min(62, radius * 0.55),
      });
    });
    if (!nodes.some((node) => String(node.object_id ?? node.id ?? '') === selectedGraphNodeId)) {
      selectedGraphNodeId = String(nodes[0]?.object_id ?? nodes[0]?.id ?? '');
    }

    svg.replaceChildren();
    relationList.replaceChildren();
    empty.hidden = nodes.length > 0;
    badge.textContent = `${nodes.length} nodes · ${edges.length} edges`;
    if (!nodes.length) {
      selectedGraphNodeId = null;
      inspector.textContent = 'No scene-graph nodes were returned for the latest frame.';
      return;
    }

    const edgeLayer = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    for (const edge of edges) {
      const sourceId = String(edge.source ?? '');
      const targetId = String(edge.target ?? '');
      const source = positionedNodes.get(sourceId);
      const target = positionedNodes.get(targetId);
      if (!source || !target || !nodeIds.has(sourceId) || !nodeIds.has(targetId)) continue;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('class', 'graph-edge');
      line.setAttribute('x1', String(source.x));
      line.setAttribute('y1', String(source.y));
      line.setAttribute('x2', String(target.x));
      line.setAttribute('y2', String(target.y));
      edgeLayer.append(line);
      const relation = edge.relationship || edge.relation;
      if (typeof relation === 'string' && relation) {
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('class', 'graph-edge-label');
        label.setAttribute('x', String((source.x + target.x) / 2));
        label.setAttribute('y', String((source.y + target.y) / 2 - 5));
        label.setAttribute('text-anchor', 'middle');
        label.textContent = relation.replaceAll('_', ' ');
        edgeLayer.append(label);
      }
    }
    svg.append(edgeLayer);

    for (const [index, node] of nodes.entries()) {
      const nodeId = String(node.object_id ?? node.id ?? '');
      if (!nodeId) continue;
      const position = positionedNodes.get(nodeId);
      const label = String(node.class_name || node.model_class_name || node.label || 'Object');
      const track = node.track_id === null || node.track_id === undefined ? '' : ` #${node.track_id}`;
      const displayLabel = `${label.slice(0, 16)}${track}`;
      const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      group.setAttribute('class', 'graph-object-node');
      group.setAttribute('data-node-id', nodeId);
      group.setAttribute('tabindex', '0');
      group.setAttribute('role', 'button');
      group.setAttribute('aria-label', `${label}${track}; select to inspect`);
      if (nodeId === selectedGraphNodeId) group.classList.add('is-selected');
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', String(position.x - 52));
      rect.setAttribute('y', String(position.y - 18));
      rect.setAttribute('width', '104');
      rect.setAttribute('height', '36');
      rect.setAttribute('rx', '10');
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', String(position.x));
      text.setAttribute('y', String(position.y + 4));
      text.setAttribute('text-anchor', 'middle');
      text.textContent = displayLabel;
      group.append(rect, text);
      const selectNode = () => {
        selectedGraphNodeId = nodeId;
        renderGraph(graph);
      };
      group.addEventListener('click', selectNode);
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          selectNode();
        }
      });
      svg.append(group);
      if (index === 0 && !nodes.some((item) => String(item.object_id ?? item.id ?? '') === selectedGraphNodeId)) {
        selectedGraphNodeId = nodeId;
      }
    }

    const selected = nodes.find((node) => String(node.object_id ?? node.id ?? '') === selectedGraphNodeId) || nodes[0];
    if (selected) {
      selectedGraphNodeId = String(selected.object_id ?? selected.id ?? '');
      const selectedName = selected.class_name || selected.model_class_name || selected.label || 'Object';
      const track = selected.track_id === null || selected.track_id === undefined ? 'not tracked' : `track #${selected.track_id}`;
      const confidence = Number.isFinite(Number(selected.confidence)) ? `${(Number(selected.confidence) * 100).toFixed(1)}% confidence` : 'confidence unavailable';
      inspector.textContent = `${selectedName} · ${track} · ${confidence} · position ${formatPosition(selected)} · relative depth ${formatRelativeDepth(selected.relative_depth_value)}`;
      for (const edge of edges.filter((item) => String(item.source ?? '') === selectedGraphNodeId || String(item.target ?? '') === selectedGraphNodeId)) {
        const item = document.createElement('li');
        const source = nodes.find((node) => String(node.object_id ?? node.id ?? '') === String(edge.source ?? ''));
        const target = nodes.find((node) => String(node.object_id ?? node.id ?? '') === String(edge.target ?? ''));
        item.textContent = `${source?.class_name || source?.label || edge.source} ${edge.relationship || edge.relation || 'relates to'} ${target?.class_name || target?.label || edge.target}`;
        relationList.append(item);
      }
    }
  }

  function renderFrame(message) {
    const objects = Array.isArray(message.objects) ? message.objects : [];
    const detections = Array.isArray(message.detections) ? message.detections : [];
    const tracks = Array.isArray(message.tracks) ? message.tracks : [];

    refreshReviewScene(message);
    window.BYAKUFrameProcessor.drawOverlay(overlay, message.frame_size, objects);
    const cameraConnected = message.stage_status?.camera === 'connected';
    detail.textContent = `${cameraConnected ? 'Live scene' : 'Last processed frame retained'} · ${tracks.length} tracked · ${message.processing_ms ?? '—'} ms per frame`;
    cameraStatus.innerHTML = cameraConnected
      ? '<span class="status-dot" aria-hidden="true"></span>Camera connected · processing'
      : '<span class="status-dot status-dot-muted" aria-hidden="true"></span>Camera disconnected · last frame available';
    updatePipeline({
      ...(message.stage_status || {}),
      backend: 'connected',
      last_frame_available: true,
      processing_ms: message.processing_ms,
      initialization_ms: message.initialization_ms,
      timing_source: message.timing_source,
    });
    renderNarration(
      message.scene_narration
      || message.scene_summary_speech?.spoken_text
      || message.scene_description?.summary,
    );

    const count = document.querySelector('#detection-count');
    if (count) count.textContent = `${detections.length} detection${detections.length === 1 ? '' : 's'}`;
    replaceRows('#detection-rows', objects, (item) => [
      item.class_name,
      item.track_id !== null && item.track_id !== undefined ? item.track_id : '—',
      Number.isFinite(Number(item.confidence)) ? `${(Number(item.confidence) * 100).toFixed(1)}%` : '—',
      formatPosition(item),
      item.depth_category || 'UNKNOWN',
      Array.isArray(item.bounding_box) ? item.bounding_box.map((value) => Math.round(value)).join(', ') : '—',
    ], 'No detections received');

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

    renderGraph(message.graph);

  }

  async function refreshStatus() {
    try {
      const response = await fetch('/api/status', { cache: 'no-store' });
      if (!response.ok) throw new Error(`Backend status returned ${response.status}`);
      const status = await response.json();
      updatePipeline(status);
      if (status.last_frame) {
        renderFrame({
          ...status.last_frame,
          frame_preview: status.last_frame_preview,
          last_frame_available: status.last_frame_available,
        });
      }
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
      button.textContent = 'Stop Demonstration';
    } catch (error) {
      detail.textContent = error.message || 'Python backend unavailable';
      cameraStatus.innerHTML = '<span class="status-dot" aria-hidden="true"></span>Backend unavailable';
      button.textContent = 'Run Live Demonstration';
      frameProcessor?.stop();
      frameProcessor = null;
      connection.close();
      if (window.BYAKUCamera.stream) {
        window.BYAKUCamera.stop(video, container, placeholder, cameraStatus, detail, null, button);
        button.textContent = 'Run Live Demonstration';
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

  document.querySelector('#review-correction-form')?.addEventListener('submit', submitCorrection);
  document.querySelector('#review-confirm')?.addEventListener('click', () => {
    if (selectedDetection) saveReview('confirm', selectedDetection.class_name, 'confirmed');
  });
  document.querySelector('#review-incorrect')?.addEventListener('click', () => {
    if (!selectedDetection) return;
    const form = document.querySelector('#review-correction-form');
    form.dataset.open = 'true';
    form.hidden = false;
    document.querySelector('#review-corrected-label').focus();
    setReviewStatus('Enter the correct object name. Nothing is saved until you confirm Save Correction.');
  });
  document.querySelector('#review-cancel')?.addEventListener('click', () => {
    const form = document.querySelector('#review-correction-form');
    form.dataset.open = 'false';
    form.reset();
    renderReviewDetections();
    setReviewStatus('Correction cancelled; no review record was saved.');
  });
  document.querySelector('#review-history-refresh')?.addEventListener('click', () => {
    loadReviewHistory().catch((error) => setReviewStatus(`Could not load review history: ${error.message}`, true));
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
  const browserSpeechStatus = document.querySelector('#system-browser-speech');
  if (browserSpeechStatus) {
    browserSpeechStatus.textContent = window.speechSynthesis && typeof window.SpeechSynthesisUtterance === 'function'
      ? 'API present · playback not reported'
      : 'API unavailable';
  }
  loadReviewHistory().catch((error) => setReviewStatus(`Could not load review history: ${error.message}`, true));
  window.setInterval(refreshStatus, 5000);
  window.addEventListener('pagehide', () => {
    frameProcessor?.stop();
    connection.close();
    if (window.BYAKUCamera.stream) {
      window.BYAKUCamera.stop(video, container, placeholder, cameraStatus, detail, null, button);
    }
  });
})();