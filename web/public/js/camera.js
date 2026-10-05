(() => {
  let activeStream = null;

  function getIdealVideoConstraints(preferredDeviceId = null) {
    const base = {
      facingMode: 'user',
      width: { ideal: 640 },
      height: { ideal: 480 },
    };
    if (preferredDeviceId) {
      return { ...base, deviceId: { exact: preferredDeviceId } };
    }
    return base;
  }

  async function getCameraDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === 'videoinput');
  }

  async function start(video, container, placeholder, status, footnote, badge, button, options = {}) {
    const deviceId = options.deviceId || null;
    const preferredFacing = options.facingMode || 'user';
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Camera access is not available in this browser.');
    }

    try {
      activeStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } } : { facingMode: preferredFacing, width: { ideal: 640 }, height: { ideal: 480 } },
      });
    } catch (error) {
      if (error?.name === 'NotAllowedError') {
        throw new Error('Camera permission was denied. Allow camera access and try again.');
      }
      if (error?.name === 'NotFoundError') {
        throw new Error('No camera device was found on this computer.');
      }
      if (error?.name === 'NotReadableError' || error?.name === 'ConstraintNotSatisfiedError') {
        throw new Error('The camera is in use by another app or cannot be opened.');
      }
      throw new Error(error?.message || 'Unable to start the camera preview.');
    }

    if (video) {
      video.srcObject = activeStream;
      video.muted = true;
      video.playsInline = true;
      video.autoplay = true;
      await video.play().catch(() => undefined);
    }

    if (container) container.classList.add('has-stream');
    if (placeholder) placeholder.setAttribute('aria-hidden', 'true');
    if (status) status.innerHTML = '<span class="status-dot" aria-hidden="true"></span>Camera preview active';
    if (footnote) footnote.textContent = 'Browser camera stream';
    if (badge) badge.hidden = false;
    if (button) button.textContent = 'Stop camera preview';

    return activeStream;
  }

  function stop(video, container, placeholder, status, footnote, badge, button) {
    activeStream?.getTracks().forEach((track) => track.stop());
    activeStream = null;
    if (video) video.srcObject = null;
    if (container) container.classList.remove('has-stream');
    if (placeholder) placeholder.removeAttribute('aria-hidden');
    if (status) status.innerHTML = '<span class="status-dot status-dot-muted" aria-hidden="true"></span>Camera Ready';
    if (footnote) footnote.textContent = 'No camera stream';
    if (badge) badge.hidden = true;
    if (button) button.textContent = 'Enable camera preview';
  }

  window.BYAKUCamera = {
    start,
    stop,
    getCameraDevices,
    get stream() { return activeStream; },
    getIdealVideoConstraints,
  };
})();
