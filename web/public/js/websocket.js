(() => {
  const STATE_SEQUENCE = ['idle', 'connecting', 'connected', 'streaming', 'paused', 'error'];

  class BYAKUConnection extends EventTarget {
    constructor(url = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`) {
      super();
      this.url = url;
      this.socket = null;
      this.state = 'idle';
      this.reconnectTimer = null;
      this.reconnectMs = 250;
      this.maxReconnectMs = 10000;
      this.heartbeatTimer = null;
      this.heartbeatMs = 25000;
      this.heartbeatAck = true;
      this.shouldReconnect = true;
      this.connectionAttempt = 0;
      this.lastMessageAt = 0;
    }

    setState(nextState) {
      if (this.state === nextState) return;
      this.state = nextState;
      this.dispatchEvent(new CustomEvent('statechange', { detail: { state: nextState } }));
    }

    static getBackoffDelay(attempt) {
      const base = 250 + Math.random() * 250;
      return Math.min(base * Math.pow(2, attempt), 10000);
    }

    scheduleReconnect() {
      if (!this.shouldReconnect) return;
      const delay = BYAKUConnection.getBackoffDelay(this.connectionAttempt);
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = window.setTimeout(() => {
        this.connectionAttempt += 1;
        this.connect();
      }, delay);
    }

    clearHeartbeat() {
      if (this.heartbeatTimer) {
        window.clearInterval(this.heartbeatTimer);
        this.heartbeatTimer = null;
      }
    }

    startHeartbeat() {
      this.clearHeartbeat();
      this.heartbeatTimer = window.setInterval(() => {
        if (this.socket?.readyState !== WebSocket.OPEN) return;
        if (!this.heartbeatAck) {
          this.socket.close();
          return;
        }
        this.heartbeatAck = false;
        try {
          this.socket.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
        } catch {
          this.socket.close();
        }
      }, this.heartbeatMs);
    }

    connect() {
      if (!this.url || !('WebSocket' in window)) {
        this.setState('error');
        return Promise.reject(new Error('WebSocket is not supported in this browser.'));
      }
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.setState('connected');
        return Promise.resolve(true);
      }

      this.setState('connecting');
      this.socket = new WebSocket(this.url);
      return new Promise((resolve, reject) => {
        const timeout = window.setTimeout(() => {
          if (this.socket && this.socket.readyState !== WebSocket.OPEN) {
            this.socket.close();
          }
          reject(new Error('Python backend connection timed out.'));
        }, 10000);

        this.socket.addEventListener('open', () => {
          window.clearTimeout(timeout);
          this.connectionAttempt = 0;
          this.heartbeatAck = true;
          this.setState('connected');
          this.startHeartbeat();
          this.dispatchEvent(new Event('connected'));
          resolve(true);
        }, { once: true });

        this.socket.addEventListener('error', (event) => {
          window.clearTimeout(timeout);
          this.setState('error');
          this.dispatchEvent(new CustomEvent('error', { detail: { message: 'Could not connect to the BYAKU backend.', raw: event } }));
          reject(event);
        }, { once: true });
      }).then(() => {
        this.socket.addEventListener('close', () => {
          this.clearHeartbeat();
          this.setState('idle');
          this.dispatchEvent(new Event('disconnected'));
          this.scheduleReconnect();
        });

        this.socket.addEventListener('message', (event) => {
          try {
            const payload = JSON.parse(event.data);
            this.lastMessageAt = Date.now();
            if (payload?.type === 'pong') {
              this.heartbeatAck = true;
              return;
            }
            if (payload?.type === 'error') {
              this.dispatchEvent(new CustomEvent('error', { detail: payload }));
            }
            this.dispatchEvent(new CustomEvent('update', { detail: payload }));
          } catch (error) {
            this.dispatchEvent(new CustomEvent('error', { detail: { type: 'error', stage: 'protocol', code: 'invalid_json', message: 'Invalid backend message', recoverable: true, raw: error } }));
          }
        });
        return true;
      });
    }

    send(payload) {
      if (!payload || this.socket?.readyState !== WebSocket.OPEN) return false;
      try {
        this.socket.send(JSON.stringify(payload));
        return true;
      } catch (error) {
        this.dispatchEvent(new CustomEvent('error', { detail: { type: 'error', stage: 'socket', code: 'send_failed', message: 'Could not send data to the backend.', recoverable: true, raw: error } }));
        return false;
      }
    }

    close() {
      this.shouldReconnect = false;
      this.clearHeartbeat();
      window.clearTimeout(this.reconnectTimer);
      if (this.socket) {
        try {
          this.socket.close();
        } catch {
          // ignore double-close
        }
      }
      this.socket = null;
      this.setState('idle');
    }

    resume() {
      this.shouldReconnect = true;
      this.connectionAttempt = 0;
      return this.connect();
    }
  }

  class BYAKUFrameProcessor extends EventTarget {
    constructor(connection, video, maxWidth = 640) {
      super();
      this.connection = connection;
      this.video = video;
      this.maxWidth = maxWidth;
      this.canvas = document.createElement('canvas');
      this.context = this.canvas.getContext('2d', { alpha: false });
      this.timer = 0;
      this.frameId = 0;
      this.busy = false;
      this.running = false;
      this.pendingFrameId = null;
      this.lastFrameSentAt = 0;
      this.frameTimeoutMs = 5000;
      this.keepAliveTimer = null;
      this.onUpdate = (event) => {
        const message = event.detail;
        if (message?.type === 'frame_result' && this.pendingFrameId && Number(message.frame_id) === this.pendingFrameId) {
          this.busy = false;
          this.pendingFrameId = null;
        } else if (message?.type === 'error' && message.stage === 'camera') {
          this.busy = false;
          this.pendingFrameId = null;
        }
        this.dispatchEvent(new CustomEvent('result', { detail: message }));
      };
    }

    start() {
      if (this.running) return;
      this.running = true;
      this.connection.addEventListener('update', this.onUpdate);
      this.connection.send({ type: 'start_camera' });
      this.connection.setState('streaming');
      this.timer = window.setInterval(() => this.capture(), 200);
    }

    stop() {
      this.running = false;
      this.busy = false;
      this.pendingFrameId = null;
      window.clearInterval(this.timer);
      this.timer = 0;
      if (this.keepAliveTimer) {
        window.clearTimeout(this.keepAliveTimer);
        this.keepAliveTimer = null;
      }
      this.connection.removeEventListener('update', this.onUpdate);
      this.connection.send({ type: 'stop_camera' });
      this.connection.setState('paused');
    }

    sendQuery(text) {
      return this.connection.send({ type: 'query', text });
    }

    async capture() {
      if (!this.running || this.busy || this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
      const sourceWidth = this.video.videoWidth;
      const sourceHeight = this.video.videoHeight;
      if (!sourceWidth || !sourceHeight) return;

      const scale = Math.min(1, this.maxWidth / sourceWidth);
      const width = Math.max(1, Math.round(sourceWidth * scale));
      const height = Math.max(1, Math.round(sourceHeight * scale));
      this.canvas.width = width;
      this.canvas.height = height;
      this.context.drawImage(this.video, 0, 0, width, height);

      try {
        const blob = await new Promise((resolve, reject) => {
          this.canvas.toBlob((value) => value ? resolve(value) : reject(new Error('Could not encode camera frame.')), 'image/jpeg', 0.7);
        });
        const dataUrl = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result).split(',')[1]);
          reader.onerror = () => reject(new Error('Could not read encoded camera frame.'));
          reader.readAsDataURL(blob);
        });

        const frameId = ++this.frameId;
        this.pendingFrameId = frameId;
        this.lastFrameSentAt = Date.now();
        this.busy = this.connection.send({
          type: 'frame',
          frame_id: frameId,
          width,
          height,
          data: dataUrl,
        });

        if (this.busy) {
          this.keepAliveTimer = window.setTimeout(() => {
            if (this.busy && this.pendingFrameId === frameId) {
              this.busy = false;
              this.pendingFrameId = null;
              this.dispatchEvent(new CustomEvent('result', { detail: { type: 'error', stage: 'camera', code: 'frame_timeout', message: 'Camera frame timed out waiting for backend result.', recoverable: true } }));
            }
          }, this.frameTimeoutMs);
        }
      } catch (error) {
        this.busy = false;
        this.pendingFrameId = null;
        this.dispatchEvent(new CustomEvent('result', { detail: { type: 'error', stage: 'camera', code: 'capture_failed', message: error.message || 'Camera frame capture failed.', recoverable: true } }));
      }
    }

    static drawOverlay(canvas, frameSize, objects) {
      const context = canvas?.getContext('2d');
      const width = canvas?.clientWidth || 0;
      const height = canvas?.clientHeight || 0;
      const sourceWidth = Number(frameSize?.width) || 0;
      const sourceHeight = Number(frameSize?.height) || 0;
      if (!context || !width || !height || !sourceWidth || !sourceHeight) return;

      const ratio = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const scale = Math.max(width / sourceWidth, height / sourceHeight);
      const offsetX = (width - sourceWidth * scale) / 2;
      const offsetY = (height - sourceHeight * scale) / 2;
      context.lineWidth = 2;
      context.font = '600 12px "DM Sans", sans-serif';
      for (const object of objects || []) {
        const box = object.bounding_box || object.bbox;
        if (!Array.isArray(box) || box.length !== 4) continue;
        const [x1, y1, x2, y2] = box.map(Number);
        if (![x1, y1, x2, y2].every(Number.isFinite)) continue;
        const left = offsetX + x1 * scale;
        const top = offsetY + y1 * scale;
        const boxWidth = (x2 - x1) * scale;
        const boxHeight = (y2 - y1) * scale;
        if (boxWidth <= 0 || boxHeight <= 0) continue;

        const distance = String(object.distance_category || '').toLowerCase();
        const colors = {
          very_close: '#ff5d73',
          close: '#ff8a3d',
          moderate: '#f7c857',
          far: '#6ecf8d',
        };
        context.strokeStyle = colors[distance] || '#d68af0';
        context.strokeRect(left, top, boxWidth, boxHeight);
        const label = [object.class_name || object.class, object.track_id != null ? `#${object.track_id}` : '', Number.isFinite(Number(object.confidence)) ? `${Math.round(Number(object.confidence) * 100)}%` : '', object.horizontal_position || object.position || '', object.distance_category || '']
          .filter(Boolean).join(' ');
        if (label) {
          const labelWidth = context.measureText(label).width + 12;
          const labelTop = Math.max(0, top - 22);
          context.fillStyle = 'rgba(50, 22, 63, .88)';
          context.fillRect(left, labelTop, labelWidth, 20);
          context.fillStyle = '#fff';
          context.fillText(label, left + 6, labelTop + 14);
        }
      }
    }
  }

  window.BYAKUConnection = BYAKUConnection;
  window.BYAKUFrameProcessor = BYAKUFrameProcessor;
})();
