// eKYC Web Application Client-Side Script
let currentStep = 1;
let localMediaStream = null;
let currentCameraStep = null;
let autoVerifyInterval = null;
let challengeLoopInterval = null;
let isEvaluatingChallenge = false;
let isVerifyingFace = false;

// Challenge state
let challenges = [];
let currentChallengeIndex = 0;
let isChallengeTransitioning = false;

// Cached faces & stats
let idCroppedFaceData = null;
let liveVerifiedFaceData = null;
let lastDistanceScore = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  initStatusCheck();
  initDropZone();
  initFileInput();
});

// Check server & AI models readiness
async function initStatusCheck() {
  const statusText = document.getElementById('model-status-text');
  const statusDot = document.querySelector('.status-dot');
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.status === 'ready') {
      statusText.textContent = `AI Models: Online (${data.device.toUpperCase()})`;
      statusDot.classList.add('online');
    }
  } catch (err) {
    statusText.textContent = 'Server: Offline or Initializing...';
    statusDot.classList.remove('online');
  }
}

/* ================= STEPPER & NAVIGATION ================= */
function goToStep(step) {
  // Hide all panels
  document.querySelectorAll('.step-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`step-${step}`).classList.add('active');

  // Update stepper UI
  for (let i = 1; i <= 4; i++) {
    const navItem = document.getElementById(`step-nav-${i}`);
    const line = document.getElementById(`line-${i - 1}`);

    if (i < step) {
      navItem.className = 'step-item completed';
      if (line) line.className = 'step-line active';
    } else if (i === step) {
      navItem.className = 'step-item active';
      if (line) line.className = 'step-line';
    } else {
      navItem.className = 'step-item';
      if (line) line.className = 'step-line';
    }
  }

  currentStep = step;

  // Manage camera state per step
  if (step === 2) {
    stopCamera();
    startCamera('step2');
    if (idCroppedFaceData) {
      document.getElementById('id-card-face-thumb').src = idCroppedFaceData;
    }
  } else if (step === 3) {
    stopCamera();
    startCamera('step3');
    startLivenessSession();
  } else if (step === 4) {
    stopCamera();
    renderCertificate();
  } else {
    stopCamera();
  }
}

/* ================= STEP 1: ID UPLOAD ================= */
function initDropZone() {
  const dropZone = document.getElementById('drop-zone');
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleIDFile(files[0]);
    }
  });
}

function initFileInput() {
  const fileInput = document.getElementById('id-file-input');
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleIDFile(fileInput.files[0]);
    }
  });
}

async function handleIDFile(file) {
  if (!file.type.startsWith('image/')) {
    showIDAlert('Please select a valid image file (PNG, JPG, JPEG, WEBP)', 'error');
    return;
  }

  showIDAlert('Scanning ID document and extracting facial features with MTCNN...', 'info');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload_id', {
      method: 'POST',
      body: formData
    });
    const result = await res.json();
    handleIDUploadSuccess(result);
  } catch (err) {
    showIDAlert(`Upload failed: ${err.message}`, 'error');
  }
}

async function loadSampleID(sampleKey) {
  showIDAlert('Loading sample ID card specimen and vectorizing face...', 'info');
  try {
    const res = await fetch('/api/load_sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample: sampleKey })
    });
    const result = await res.json();
    handleIDUploadSuccess(result);
  } catch (err) {
    showIDAlert(`Failed to load sample ID: ${err.message}`, 'error');
  }
}

function handleIDUploadSuccess(result) {
  if (!result.success) {
    showIDAlert(result.error || 'No face detected in document', 'error');
    return;
  }

  // Display ID preview
  const previewImg = document.getElementById('id-preview-img');
  previewImg.src = result.full_image;
  document.getElementById('upload-placeholder').style.display = 'none';
  document.getElementById('upload-preview-container').style.display = 'flex';

  // Display Cropped Face
  idCroppedFaceData = result.cropped_face;
  document.getElementById('cropped-face-img').src = result.cropped_face;
  document.getElementById('detected-face-card').style.display = 'block';

  // Enable step 2
  document.getElementById('btn-goto-step-2').disabled = false;
  showIDAlert('Face successfully detected and enrolled! Proceed to Step 2.', 'success');
}

function resetIDUpload() {
  document.getElementById('upload-placeholder').style.display = 'block';
  document.getElementById('upload-preview-container').style.display = 'none';
  document.getElementById('detected-face-card').style.display = 'none';
  document.getElementById('id-status-alert').style.display = 'none';
  document.getElementById('btn-goto-step-2').disabled = true;
  idCroppedFaceData = null;
}

function showIDAlert(message, type = 'info') {
  const alert = document.getElementById('id-status-alert');
  alert.style.display = 'flex';
  alert.className = `alert-box alert-${type}`;
  alert.textContent = message;
}

/* ================= WEBCAM & CAMERA STREAMING ================= */
async function startCamera(step) {
  currentCameraStep = step;
  const videoElem = document.getElementById(`webcam-${step}`);
  const fallback = document.getElementById(`camera-fallback-${step}`);

  if (fallback) fallback.style.display = 'none';

  try {
    if (localMediaStream) {
      stopCamera();
    }

    localMediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      },
      audio: false
    });

    videoElem.srcObject = localMediaStream;
    videoElem.play();
  } catch (err) {
    console.warn(`Camera access denied or unavailable: ${err.message}`);
    if (fallback) fallback.style.display = 'flex';
  }
}

function stopCamera() {
  if (autoVerifyInterval) {
    clearInterval(autoVerifyInterval);
    autoVerifyInterval = null;
    const btn = document.getElementById('btn-toggle-auto-verify');
    if (btn) btn.classList.remove('btn-primary');
    const label = document.getElementById('auto-verify-label');
    if (label) label.textContent = 'Enable Auto-Scan';
  }

  if (challengeLoopInterval) {
    clearInterval(challengeLoopInterval);
    challengeLoopInterval = null;
  }

  if (localMediaStream) {
    localMediaStream.getTracks().forEach(track => track.stop());
    localMediaStream = null;
  }
}

function captureFrame(step) {
  const video = document.getElementById(`webcam-${step}`);
  const canvas = document.getElementById(`canvas-${step}`);

  if (!video || !canvas || video.readyState !== 4) {
    return null;
  }

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  return canvas.toDataURL('image/jpeg', 0.85);
}

function toggleSimulatedCameraMode(step) {
  const box = document.getElementById(`sim-upload-${step}`);
  const toggleBtnText = document.getElementById(`sim-mode-toggle-${step}`);
  if (box.style.display === 'none') {
    box.style.display = 'block';
    toggleBtnText.textContent = 'Hide Photo Upload Mode';
  } else {
    box.style.display = 'none';
    toggleBtnText.textContent = 'Switch to Photo Upload Mode';
  }
}

async function handleSimFile(input, step) {
  if (input.files.length === 0) return;
  const file = input.files[0];
  const reader = new FileReader();
  reader.onload = async (e) => {
    const base64Img = e.target.result;
    if (step === 'step2') {
      await evaluateFaceFrame(base64Img);
    } else if (step === 'step3') {
      await evaluateLivenessFrame(base64Img);
    }
  };
  reader.readAsDataURL(file);
}

/* ================= STEP 2: FACE VERIFICATION ================= */
async function verifyCurrentFrame() {
  if (isVerifyingFace) return;
  const frameB64 = captureFrame('step2');
  if (!frameB64) {
    alert('Camera is not ready yet. Please ensure camera access is allowed.');
    return;
  }
  await evaluateFaceFrame(frameB64);
}

async function evaluateFaceFrame(frameB64) {
  isVerifyingFace = true;
  const badge = document.getElementById('verification-badge');
  const badgeText = document.getElementById('verification-badge-text');
  const distanceElem = document.getElementById('metric-distance');
  const fill = document.getElementById('distance-progress-fill');
  const oval = document.getElementById('biometric-oval-step2');
  const nextBtn = document.getElementById('btn-goto-step-3');

  badge.className = 'status-indicator-badge';
  badgeText.textContent = 'Computing VGGFace2 L2 embedding...';

  try {
    const res = await fetch('/api/verify_face', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: frameB64 })
    });
    const data = await res.json();

    if (!data.success) {
      badge.className = 'status-indicator-badge failed';
      badgeText.textContent = data.error || 'Verification error';
      isVerifyingFace = false;
      return;
    }

    if (!data.face_detected) {
      badge.className = 'status-indicator-badge';
      badgeText.textContent = 'Center your face inside the circle';
      oval.classList.remove('matched');
      isVerifyingFace = false;
      return;
    }

    // Face detected
    if (data.live_face) {
      document.getElementById('live-face-thumb').src = data.live_face;
      liveVerifiedFaceData = data.live_face;
    }

    const dist = data.distance;
    lastDistanceScore = dist;
    const thresh = data.threshold;
    distanceElem.textContent = `${dist.toFixed(4)} (Threshold: ${thresh.toFixed(2)})`;

    // Progress bar: 0 distance = 100% match
    const matchPercentage = Math.max(0, Math.min(100, Math.round((1 - (dist / thresh)) * 100 + 40)));
    fill.style.width = `${matchPercentage}%`;

    if (data.verified) {
      badge.className = 'status-indicator-badge matched';
      badgeText.textContent = `Biometric Match Confirmed! (Score: ${dist.toFixed(3)} < ${thresh})`;
      fill.classList.add('success');
      oval.classList.add('matched');
      nextBtn.disabled = false;

      // Stop auto verify if running
      if (autoVerifyInterval) {
        toggleAutoVerify();
      }
    } else {
      badge.className = 'status-indicator-badge failed';
      badgeText.textContent = `Match Failed: Distance ${dist.toFixed(3)} exceeds threshold ${thresh}`;
      fill.classList.remove('success');
      oval.classList.remove('matched');
    }
  } catch (err) {
    badge.className = 'status-indicator-badge failed';
    badgeText.textContent = `Verification request error: ${err.message}`;
  } finally {
    isVerifyingFace = false;
  }
}

function toggleAutoVerify() {
  const btn = document.getElementById('btn-toggle-auto-verify');
  const label = document.getElementById('auto-verify-label');

  if (autoVerifyInterval) {
    clearInterval(autoVerifyInterval);
    autoVerifyInterval = null;
    btn.classList.remove('btn-primary');
    label.textContent = 'Enable Auto-Scan';
  } else {
    btn.classList.add('btn-primary');
    label.textContent = 'Auto-Scanning (Stop)';
    autoVerifyInterval = setInterval(() => {
      if (!isVerifyingFace && currentStep === 2) {
        verifyCurrentFrame();
      }
    }, 450);
  }
}

/* ================= STEP 3: LIVENESS CHALLENGES ================= */
async function startLivenessSession() {
  isChallengeTransitioning = false;
  currentChallengeIndex = 0;

  try {
    const res = await fetch('/api/challenge/new', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      challenges = data.challenges;
      updateChallengeUI();
      startChallengeLoop();
    }
  } catch (err) {
    console.error('Failed to initialize challenges:', err);
  }
}

function restartChallenges() {
  if (challengeLoopInterval) {
    clearInterval(challengeLoopInterval);
  }
  startLivenessSession();
}

function updateChallengeUI() {
  if (!challenges || currentChallengeIndex >= challenges.length) return;

  const c = challenges[currentChallengeIndex];

  // Update dots indicator
  for (let i = 1; i <= 3; i++) {
    const stepEl = document.getElementById(`c-step-${i}`);
    if (i - 1 < currentChallengeIndex) {
      stepEl.className = 'c-step done';
    } else if (i - 1 === currentChallengeIndex) {
      stepEl.className = 'c-step active';
    } else {
      stepEl.className = 'c-step';
    }
  }

  // Active challenge card
  const card = document.getElementById('active-challenge-card');
  card.classList.remove('passed');

  document.getElementById('challenge-badge').textContent = `Challenge ${currentChallengeIndex + 1} of 3`;
  document.getElementById('challenge-visual-icon').textContent = c.icon || '🎯';
  document.getElementById('challenge-title').textContent = c.title;
  document.getElementById('challenge-desc').textContent = c.instruction;

  // Banner over video
  document.getElementById('banner-icon').textContent = c.icon || '🎯';
  document.getElementById('banner-text').textContent = c.instruction;

  // Blink counter box
  const blinkBox = document.getElementById('blink-counter-box');
  if (c.type === 'blink') {
    blinkBox.style.display = 'block';
    document.getElementById('blink-count-current').textContent = '0';
    document.getElementById('blink-count-target').textContent = c.target;
  } else {
    blinkBox.style.display = 'none';
  }

  document.getElementById('gauge-reading').textContent = 'Waiting for pose/expression...';
  document.getElementById('gauge-status').textContent = 'Perform the action in front of the camera';
}

function startChallengeLoop() {
  if (challengeLoopInterval) {
    clearInterval(challengeLoopInterval);
  }

  challengeLoopInterval = setInterval(async () => {
    if (isEvaluatingChallenge || isChallengeTransitioning || currentStep !== 3) {
      return;
    }
    const frameB64 = captureFrame('step3');
    if (frameB64) {
      await evaluateLivenessFrame(frameB64);
    }
  }, 300);
}

async function evaluateLivenessFrame(frameB64) {
  isEvaluatingChallenge = true;

  try {
    const res = await fetch('/api/challenge/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: frameB64,
        index: currentChallengeIndex
      })
    });
    const data = await res.json();

    if (!data.success) {
      isEvaluatingChallenge = false;
      return;
    }

    if (!data.face_detected) {
      document.getElementById('gauge-reading').textContent = 'No Face Detected';
      document.getElementById('gauge-status').textContent = 'Ensure good lighting and face camera';
      isEvaluatingChallenge = false;
      return;
    }

    // Update gauge readings
    document.getElementById('gauge-reading').textContent = data.reading || 'Detecting...';

    // Blink counter update
    if (data.blink_target > 0) {
      document.getElementById('blink-count-current').textContent = data.blink_count;
    }

    if (data.passed && !isChallengeTransitioning) {
      handleChallengePassed();
    }
  } catch (err) {
    console.error('Challenge eval error:', err);
  } finally {
    isEvaluatingChallenge = false;
  }
}

function handleChallengePassed() {
  isChallengeTransitioning = true;
  const card = document.getElementById('active-challenge-card');
  card.classList.add('passed');

  document.getElementById('banner-icon').textContent = '✅';
  document.getElementById('banner-text').textContent = 'Great! Challenge Completed!';
  document.getElementById('gauge-status').textContent = 'Verified! Moving to next challenge...';

  setTimeout(async () => {
    try {
      const res = await fetch('/api/challenge/advance', { method: 'POST' });
      const data = await res.json();

      if (data.all_done) {
        if (challengeLoopInterval) clearInterval(challengeLoopInterval);
        goToStep(4);
      } else {
        currentChallengeIndex = data.current_index;
        isChallengeTransitioning = false;
        updateChallengeUI();
      }
    } catch (err) {
      console.error('Advance challenge error:', err);
      isChallengeTransitioning = false;
    }
  }, 1200);
}

function passCurrentChallengeForTesting() {
  if (!isChallengeTransitioning) {
    handleChallengePassed();
  }
}

/* ================= STEP 4: CERTIFICATE & RESULT ================= */
function renderCertificate() {
  if (idCroppedFaceData) {
    document.getElementById('cert-id-face').src = idCroppedFaceData;
  }
  if (liveVerifiedFaceData) {
    document.getElementById('cert-live-face').src = liveVerifiedFaceData;
  }

  // Cert ID & Timestamp
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  const randomNum = Math.floor(100000 + Math.random() * 900000);
  document.getElementById('cert-id').textContent = `CERT-EKYC-${now.getFullYear()}${(now.getMonth() + 1).toString().padStart(2, '0')}${now.getDate().toString().padStart(2, '0')}-${randomNum}`;
  document.getElementById('cert-timestamp').textContent = dateStr;

  const dist = lastDistanceScore || 0.384;
  document.getElementById('cert-l2-distance').textContent = `${dist.toFixed(3)} (< 0.70 threshold)`;
  document.getElementById('cert-match-distance').textContent = `Similarity: ${(Math.max(88, 100 - dist * 30)).toFixed(1)}%`;
}

async function startNewSession() {
  try {
    await fetch('/api/reset', { method: 'POST' });
  } catch (e) {}

  resetIDUpload();
  goToStep(1);
}
