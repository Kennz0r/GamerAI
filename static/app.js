function App() {
  
  const [conversation, setConversation] = React.useState([]);
  const [pending, setPending] = React.useState('');
  const [recording, setRecording] = React.useState(false);
  const [textChannelId, setTextChannelId] = React.useState('');
  const [userName, setUserName] = React.useState('');
  const [userText, setUserText] = React.useState('');
  const [speechEnabled, setSpeechEnabled] = React.useState(true);
  const [discordEnabled, setDiscordEnabled] = React.useState(() => {
    const stored = localStorage.getItem('discordEnabled');
    return stored !== null ? JSON.parse(stored) : true;
  });
  const [ttsEnabled, setTtsEnabled] = React.useState(() => {
    const stored = localStorage.getItem('ttsEnabled');
    return stored !== null ? JSON.parse(stored) : true;
  });
  const [botRunning, setBotRunning] = React.useState(false);
  const [timings, setTimings] = React.useState({ speech_ms: 0, llm_ms: 0, tts_ms: 0, total_ms: 0 });
  const [log, setLog] = React.useState('');
  const [showLog, setShowLog] = React.useState(false);
  const [screenshot, setScreenshot] = React.useState(null);
  const canvasRef = React.useRef(null);
  const imgRef = React.useRef(null);
  const dragRef = React.useRef(null);
  const [crop, setCrop] = React.useState(null);
  const [piperSettings, setPiperSettings] = React.useState({
    voice: '',
    rate: '1.0',
    pitch_st: '0',
    atempo: '1.0',
    length_scale: '0.95',
    noise_scale: '0.5',
    noise_w: '0.7'
  });
  const mediaRecorderRef = React.useRef(null);
  const chunksRef = React.useRef([]);
  const conversationEndRef = React.useRef(null);

  const [watchMode, setWatchMode] = React.useState('change'); // 'interval' | 'change'
  const [watchSensitivity, setWatchSensitivity] = React.useState(35); // 5..95 (percent pixels changed)
  const [watchMinGapMs, setWatchMinGapMs] = React.useState(10000); // at least 10s between sends
  const lastTinyRef = React.useRef(null);    // Uint8Array of last tiny grayscale
  const lastSentAtRef = React.useRef(0);

  const [watching, setWatching] = React.useState(false);
  const [watcherStatus, setWatcherStatus] = React.useState('Idle');
  const [watcherPrompt, setWatcherPrompt] = React.useState('Se på bildet sporadisk og sleng en kort kommentar tild et du ser).');

  const watcherVideoRef = React.useRef(null);
  const watcherCanvasRef = React.useRef(null);
  const watcherTimerRef = React.useRef(null);
  const watcherStreamRef = React.useRef(null);

    // Passive screen grab + auto-attach
  const [attachLatestFrame, setAttachLatestFrame] = React.useState(true);
  const [screenSourceActive, setScreenSourceActive] = React.useState(false);
  const [screenStatus, setScreenStatus] = React.useState('No screen selected');

  const latestFrameRef = React.useRef(null);
  const grabberVideoRef = React.useRef(null);
  const grabberCanvasRef = React.useRef(null);
  const grabberTimerRef = React.useRef(null);
  const grabberStreamRef = React.useRef(null);
  const lastVisionSentRef = React.useRef(0);

  const [imagePolicy, setImagePolicy] = React.useState('auto'); // 'auto' | 'always' | 'never'
  const [latestFrameUrl, setLatestFrameUrl] = React.useState(null);
  const previewTickRef = React.useRef(0);



  const statusClass = enabled => `status-button ${enabled ? 'on' : 'off'}`;

  React.useEffect(() => {
    const fetchData = () => {
      fetch('/conversation')
        .then(res => res.json())
        .then(data => setConversation(data));
      fetch('/pending_message')
        .then(res => res.json())
        .then(data => setPending(data.reply || ''));
      fetch('/speech_recognition')
        .then(res => res.json())
        .then(data => setSpeechEnabled(data.enabled));
      fetch('/discord_send')
        .then(res => res.json())
        .then(data => {
          setDiscordEnabled(data.enabled);
          localStorage.setItem('discordEnabled', JSON.stringify(data.enabled));
        });
      fetch('/discord_bot')
        .then(res => res.json())
        .then(data => setBotRunning(data.running));
      fetch('/log')
        .then(res => res.json())
        .then(data => setLog(data.log));
      fetch('/piper_settings')
        .then(res => res.json())
        .then(data => setPiperSettings(data));
      fetch('/timings')
        .then(res => res.json())
        .then(data => setTimings(data));
      fetch('/image_policy')
        .then(r => r.json())
        .then(d => setImagePolicy(d.mode || 'auto'))
        .catch(() => {});
    };
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  React.useEffect(() => {
    if (!ttsEnabled) return;
    const interval = setInterval(async () => {
      const res = await fetch('/tts_preview');
      if (res.status === 200) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.play();
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [ttsEnabled]);



  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);
    mediaRecorderRef.current.ondataavailable = e => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    mediaRecorderRef.current.onstop = async () => {
      const file = new File(chunksRef.current, 'audio.webm', { type: 'audio/webm' });
      chunksRef.current = [];
      const formData = new FormData();
      formData.append('file', file);
      if (textChannelId) formData.append('channel_id', textChannelId);
      formData.append('user_name', userName);
      if (screenshot) {
  const img = getCroppedImage();
  if (img) formData.append('image', img);
} else if (attachLatestFrame && latestFrameRef.current) {
  formData.append('image', latestFrameRef.current);
}
      await fetch('/queue_audio', { method: 'POST', body: formData });
    };
    mediaRecorderRef.current.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  const captureScreen = async () => {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
      const track = stream.getVideoTracks()[0];
      const imageCapture = new ImageCapture(track);
      const bitmap = await imageCapture.grabFrame();
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext('2d').drawImage(bitmap, 0, 0);
      track.stop();
      const url = canvas.toDataURL('image/png');
      setScreenshot(url);
    } catch (err) {
      console.error('Screen capture failed', err);
    }
  };

  async function setServerImagePolicy(mode) {
  try {
    const res = await fetch('/image_policy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    });
    const j = await res.json();
    setImagePolicy(j.mode || mode);
  } catch (e) {
    console.warn('Failed to set image policy:', e);
    setImagePolicy(mode); // optimistic UI
  }
}


  React.useEffect(() => {
    if (!screenshot) return;
    const img = new Image();
    img.src = screenshot;
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = img.width;
      canvas.height = img.height;
      canvas.getContext('2d').drawImage(img, 0, 0);
      imgRef.current = img;
    };
  }, [screenshot]);

  const startCrop = e => {
    if (!imgRef.current) return;
    const rect = e.target.getBoundingClientRect();
    dragRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };
  const moveCrop = e => {
    if (!dragRef.current || !imgRef.current) return;
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgRef.current, 0, 0);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.strokeRect(dragRef.current.x, dragRef.current.y, x - dragRef.current.x, y - dragRef.current.y);
  };
  const endCrop = e => {
    if (!dragRef.current) return;
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const startX = dragRef.current.x;
    const startY = dragRef.current.y;
    setCrop({ x: Math.min(startX, x), y: Math.min(startY, y), w: Math.abs(x - startX), h: Math.abs(y - startY) });
    dragRef.current = null;
  };

  const getCroppedImage = () => {
    if (!imgRef.current) return null;
    const img = imgRef.current;
    const c = crop || { x: 0, y: 0, w: img.width, h: img.height };
    const out = document.createElement('canvas');
    out.width = c.w;
    out.height = c.h;
    out.getContext('2d').drawImage(img, c.x, c.y, c.w, c.h, 0, 0, c.w, c.h);
    return out.toDataURL('image/png');
  };

  const sendText = async (e) => {
    e.preventDefault();
    const payload = {
      user_message: userText,
      user_name: userName,
    };
    if (textChannelId) payload.channel_id = textChannelId;
    if (screenshot) {
  const img = getCroppedImage();
  if (img) payload.image = img;
} else if (attachLatestFrame && latestFrameRef.current) {
  payload.image = latestFrameRef.current;
}
    await fetch('/queue', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    setUserText('');
    setScreenshot(null);
    setCrop(null);
  };

  const updateSpeechEnabled = async enabled => {
    await fetch('/speech_recognition', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    setSpeechEnabled(enabled);
  };

  const updateDiscordEnabled = async enabled => {
    const res = await fetch('/discord_send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    let data;
    try {
      data = await res.json();
    } catch (e) {
      data = { enabled };
    }
    setDiscordEnabled(!!data.enabled);
    localStorage.setItem('discordEnabled', JSON.stringify(!!data.enabled));
  };

  const toggleTts = () => {
    const val = !ttsEnabled;
    setTtsEnabled(val);
    localStorage.setItem('ttsEnabled', JSON.stringify(val));
  };

  const updatePiperSetting = (field, value) => {
    setPiperSettings(ps => ({ ...ps, [field]: value }));
  };

  const savePiperSettings = async () => {
    await fetch('/piper_settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(piperSettings),
    });
  };

  const toggleBot = async () => {
  const action = botRunning ? 'stop' : 'start';
  const res = await fetch('/discord_bot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    });
    let data = {};
    try {
      data = await res.json();
    } catch (e) {}
    if (data.status === 'started' || data.status === 'already_running') {
      setBotRunning(true);
    } else if (data.status === 'stopped' || data.status === 'not_running') {
      setBotRunning(false);
    }
  };

  

  const sendPendingToDiscord = async e => {
    e.preventDefault();
    if (!discordEnabled) return;
    await fetch('/approve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ reply: pending }).toString(),
    });
    setPending('');
  };

  const clearPending = async () => {
    await fetch('/pending_message', { method: 'DELETE' });
    setPending('');
  };

  const rate = async (index, rating) => {
    await fetch('/rate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index, rating }),
    });
    const res = await fetch('/conversation');
    const data = await res.json();
    setConversation(data);
  };

  const sendConversationToTraining = async () => {
    const res = await fetch('/conversation_training', { method: 'POST' });
    const data = await res.json();
    alert(JSON.stringify(data));
  };

    async function startWatcher() {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 5 }, audio: false
      });
      watcherStreamRef.current = stream;

      // auto-stop when sharing ends
      stream.getVideoTracks()[0].addEventListener('ended', stopWatcher);

      // hidden video to draw frames from
      if (!watcherVideoRef.current) {
        const v = document.createElement('video');
        v.muted = true;
        v.playsInline = true;
        watcherVideoRef.current = v;
      }
      watcherVideoRef.current.srcObject = stream;
      await watcherVideoRef.current.play();

      if (!watcherCanvasRef.current) {
        watcherCanvasRef.current = document.createElement('canvas');
      }

      setWatching(true);
      setWatcherStatus('Watching…');

      // tick every 1.5s
      watcherTimerRef.current = setInterval(captureAndSendFrame, 500);
      document.addEventListener('visibilitychange', onWatcherVisibility, { passive: true });
    } catch (e) {
      console.error('Screen share denied/failed:', e);
      setWatcherStatus('Permission denied or failed');
    }
  }

  function stopWatcher() {
    if (watcherTimerRef.current) { clearInterval(watcherTimerRef.current); watcherTimerRef.current = null; }
    document.removeEventListener('visibilitychange', onWatcherVisibility);

    const v = watcherVideoRef.current;
    if (v) v.srcObject = null;

    if (watcherStreamRef.current) {
      watcherStreamRef.current.getTracks().forEach(t => t.stop());
      watcherStreamRef.current = null;
    }
    setWatching(false);
    setWatcherStatus('Stopped');
  }

  function onWatcherVisibility() {
    if (document.visibilityState === 'hidden') {
      if (watcherTimerRef.current) { clearInterval(watcherTimerRef.current); watcherTimerRef.current = null; }
      setWatcherStatus('Paused (tab hidden)');
    } else if (document.visibilityState === 'visible' && watching && !watcherTimerRef.current) {
      setWatcherStatus('Watching…');
      watcherTimerRef.current = setInterval(captureAndSendFrame, 1500);
    }
  }

    async function captureAndSendFrame() {
    try {
      const v = watcherVideoRef.current;
      if (!v || v.readyState < 2) return;

      // Draw full frame (downscaled) for sending
      const maxW = 1280;
      const scale = Math.min(1, maxW / (v.videoWidth || maxW));
      const w = Math.max(1, Math.floor((v.videoWidth || maxW) * scale));
      const h = Math.max(1, Math.floor((v.videoHeight || Math.round(maxW * 9/16)) * scale));

      const c = watcherCanvasRef.current;
      c.width = w; c.height = h;
      const ctx = c.getContext('2d');
      ctx.drawImage(v, 0, 0, w, h);

      // --- Lightweight change detection on a tiny grayscale thumbnail ---
      // Downscale to tiny canvas to compute percent of changed pixels.
      const TW = 96, TH = Math.max(1, Math.round((h / w) * 96));
      const tc = document.createElement('canvas');
      tc.width = TW; tc.height = TH;
      const tctx = tc.getContext('2d');
      tctx.drawImage(c, 0, 0, TW, TH);
      const imgData = tctx.getImageData(0, 0, TW, TH).data;

      // Convert to grayscale Uint8
      const gray = new Uint8Array(TW * TH);
      for (let i = 0, j = 0; i < imgData.length; i += 4, j++) {
        // luma approximation
        gray[j] = (imgData[i] * 299 + imgData[i+1] * 587 + imgData[i+2] * 114) / 1000;
      }

      let shouldSend = false;
      if (watchMode === 'interval') {
        // Only respect the min-gap
        shouldSend = (performance.now() - lastSentAtRef.current) >= watchMinGapMs;
      } else {
        // 'change' mode: compute percent of pixels whose absolute diff > threshold
        // threshold auto-scales from sensitivity (lower sensitivity => easier to trigger)
        const last = lastTinyRef.current;
        if (!last) {
          shouldSend = true; // first frame
        } else {
          // Pick a per-pixel threshold from sensitivity (5..95 -> 40..8)
          const perPixelThresh = Math.round(48 - (watchSensitivity * 0.42)); // ~8..48
          let changed = 0;
          for (let k = 0; k < gray.length; k++) {
            if (Math.abs(gray[k] - last[k]) > perPixelThresh) changed++;
          }
          const pct = (changed / gray.length) * 100;
          // Require both sufficient change AND min-gap since last send
          if (pct >= watchSensitivity && (performance.now() - lastSentAtRef.current) >= watchMinGapMs) {
            shouldSend = true;
          }
        }
      }
      // update last tiny always
      lastTinyRef.current = gray;

      if (!shouldSend) {
        setWatcherStatus('No significant change');
        return;
      }

      // JPEG compress for speed
      const dataUrl = c.toDataURL('image/jpeg', 0.7);

      const body = {
        user_message: watcherPrompt,
        user_name: userName || 'ScreenWatcher',
        image: dataUrl
      };
      if (textChannelId) body.channel_id = textChannelId;

      const res = await fetch('/queue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!res.ok) {
        setWatcherStatus(`POST failed: ${res.status}`);
        return;
      }
      lastSentAtRef.current = performance.now();
      setWatcherStatus('Sent frame');
    } catch (e) {
      console.error('capture/send error:', e);
      setWatcherStatus('Error sending');
    }
  }

  async function selectScreenSource() {
  try {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 5 }, audio: false
    });
    grabberStreamRef.current = stream;
    stream.getVideoTracks()[0].addEventListener('ended', stopScreenSource);

    if (!grabberVideoRef.current) {
      const v = document.createElement('video');
      v.muted = true; v.playsInline = true;
      grabberVideoRef.current = v;
    }
    grabberVideoRef.current.srcObject = stream;
    await grabberVideoRef.current.play();

    if (!grabberCanvasRef.current) {
      grabberCanvasRef.current = document.createElement('canvas');
    }

    setScreenSourceActive(true);
    setScreenStatus('Capturing…');

    // keep a fresh frame in memory (no POSTs)
    grabberTimerRef.current = setInterval(captureLatestFrame, 500);
  } catch (e) {
    console.error('selectScreenSource failed:', e);
    setScreenStatus('Permission denied or failed');
  }
}

async function grabScreenAsDataURL() {
  const video = document.querySelector('#screen, #preview, video');
  if (!video || video.readyState < 2) return null;
  const c = document.createElement('canvas');
  c.width = video.videoWidth || 1280;
  c.height = video.videoHeight || 720;
  c.getContext('2d').drawImage(video, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.85);
}

setInterval(async () => {
  try {
    const r = await fetch('/vision/request', { cache: 'no-store' });
    const req = await r.json();
    if (!req || (!req.channel_id && !req.guild_id)) return;

    // make sure we actually have a frame
    const img = (typeof latestFrameRef !== 'undefined' && latestFrameRef.current) ? latestFrameRef.current : null;
    if (!img) return;

    await fetch('/vision/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        channel_id: req.channel_id || null,
        guild_id: req.guild_id || null,   // include guild to avoid lookup delay
        image: img
      })
    });
    // optional: console.log('[VISION] pushed frame on request');
  } catch (e) {
    // silent
  }
}, 800); // respond fast

function captureLatestFrame() {
  const v = grabberVideoRef.current;
  if (!v || v.readyState < 2) return;

  const maxW = 960; // smaller == lighter payload when used
  const scale = Math.min(1, maxW / (v.videoWidth || maxW));
  const w = Math.max(1, Math.floor((v.videoWidth || maxW) * scale));
  const h = Math.max(1, Math.floor((v.videoHeight || Math.round(maxW * 9/16)) * scale));

  const c = grabberCanvasRef.current;
  c.width = w; c.height = h;
  c.getContext('2d').drawImage(v, 0, 0, w, h);

  // store as data URL (your backend already accepts these)
  latestFrameRef.current = c.toDataURL('image/jpeg', 0.6);
  // Throttle preview updates a bit to avoid excessive re-renders
  const now = performance.now();
  if (now - (previewTickRef.current || 0) > 500) {
    setLatestFrameUrl(latestFrameRef.current);
    previewTickRef.current = now;
  }

  // Periodically push the latest frame to the backend so Discord users get vision
  const nowMs = Date.now();
  if (textChannelId && nowMs - lastVisionSentRef.current > 5000) {
    lastVisionSentRef.current = nowMs;
    const payload = { image: latestFrameRef.current, channel_id: textChannelId };
    fetch('/vision/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).catch(() => {});
  }
}

function stopScreenSource() {
  if (grabberTimerRef.current) { clearInterval(grabberTimerRef.current); grabberTimerRef.current = null; }
  if (grabberVideoRef.current) grabberVideoRef.current.srcObject = null;
  if (grabberStreamRef.current) {
    grabberStreamRef.current.getTracks().forEach(t => t.stop());
    grabberStreamRef.current = null;
  }
  setScreenSourceActive(false);
  setScreenStatus('Stopped');
}

// stop grabber if the component unmounts
React.useEffect(() => () => stopScreenSource(), []);



  // cleanup on unmount
  React.useEffect(() => {
    return () => stopWatcher();
  }, []);


  return (
    <div className="container">
      <div className="left-panel">
        <h2>Conversation</h2>
        <div className="conversation">
          {conversation.map((c, i) => (
            <p key={i}>
              <b>{c.user_name}:</b> {c.user_message}
              <br />
              <b>Arne:</b> {c.reply}
              <br />
              <span className="rating-buttons">
                <button onClick={() => rate(i, 'up')} disabled={c.rating === 'up'}>👍</button>
                <button onClick={() => rate(i, 'down')} disabled={c.rating === 'down'}>👎</button>
              </span>
            </p>
          ))}
          <div ref={conversationEndRef}></div>
        </div>
        <button onClick={sendConversationToTraining}>Use Conversation for Training</button>
        <h2>Send Message</h2>
        <form onSubmit={sendText}>
          <input
            type="text"
            value={userName}
            onChange={e => setUserName(e.target.value)}
            placeholder="User Name"
          /><br />
          <input
            type="text"
            value={textChannelId}
            onChange={e => setTextChannelId(e.target.value)}
            placeholder="Text Channel ID"
          /><br />
          <textarea
            rows="2"
            cols="50"
            value={userText}
            onChange={e => setUserText(e.target.value)}
            placeholder="Message"
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                sendText(e);
              }
            }}
          ></textarea><br />
          {screenshot && (
            <div>
              <canvas
                ref={canvasRef}
                onMouseDown={startCrop}
                onMouseMove={moveCrop}
                onMouseUp={endCrop}
                style={{ maxWidth: '100%', border: '1px solid #ccc' }}
              ></canvas>
            </div>
          )}
          <h3>Screen Watcher</h3>
          <label>
            <div style={{marginBottom: 6}}>Prompt sent with each frame:</div>
            <textarea
              rows="2"
              cols="50"
              value={watcherPrompt}
              onChange={e => setWatcherPrompt(e.target.value)}
              placeholder="What should the AI do each time it sees a new frame?"
            />
          </label>
          <button type="button" onClick={captureScreen}>Capture Screen</button>
          <div className="send-controls" style={{ gap: 8 }}>
            <button
              type="button"
              className={statusClass(watching)}
              onClick={watching ? stopWatcher : startWatcher}
            >
              {watching ? 'Stop Screen Watcher' : 'Start Screen Watcher'}
            </button>
            <span style={{ alignSelf: 'center', fontSize: 13, opacity: 0.8 }}>
              {watcherStatus}
            </span>
          </div>
          
          <h3>Live Screen Attachment</h3>
<div className="send-controls" style={{ gap: 8 }}>
  <button
    type="button"
    className={statusClass(screenSourceActive)}
    onClick={screenSourceActive ? stopScreenSource : selectScreenSource}
  >
    {screenSourceActive ? 'Stop Screen Source' : 'Select Screen Source'}
  </button>

  <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
    <input
      type="checkbox"
      checked={attachLatestFrame}
      onChange={e => setAttachLatestFrame(e.target.checked)}
    />
    Attach latest frame to voice/text
  </label>

  <span style={{ alignSelf: 'center', fontSize: 13, opacity: 0.8 }}>
    {screenStatus}
  </span>
</div>

{/* Image policy control (dropdown) */}
<div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 8 }}>
  <label style={{ fontSize: 13 }}>
    Image policy:&nbsp;
    <select
      value={imagePolicy}
      onChange={e => setServerImagePolicy(e.target.value)}
    >
      <option value="auto">Only when I ask (auto)</option>
      <option value="always">Always attach (if available)</option>
      <option value="never">Never attach</option>
    </select>
  </label>
</div>

{/* Live preview of the passive grabber frame */}
<div style={{ marginTop: 10 }}>
  <div style={{ fontSize: 13, opacity: 0.75, marginBottom: 6 }}>
    Latest captured frame {latestFrameUrl ? '' : '(no source yet)'}
  </div>
  {latestFrameUrl && (
    <img
      src={latestFrameUrl}
      alt="Latest screen frame"
      style={{ maxWidth: '100%', border: '1px solid #ccc', borderRadius: 8 }}
    />
  )}
</div>


                    <div style={{ display: 'grid', gap: 6, marginTop: 8 }}>
            <label style={{ fontSize: 13 }}>
              Mode:&nbsp;
              <select value={watchMode} onChange={e => setWatchMode(e.target.value)}>
                <option value="change">On change</option>
                <option value="interval">Interval only</option>
              </select>
            </label>

            {watchMode === 'change' && (
              <label style={{ fontSize: 13 }}>
                Sensitivity ({watchSensitivity}% pixels must change):
                <input
                  type="range" min="5" max="95" step="1"
                  value={watchSensitivity}
                  onChange={e => setWatchSensitivity(parseInt(e.target.value, 10))}
                  style={{ width: '100%' }}
                />
              </label>
            )}

            <label style={{ fontSize: 13 }}>
              Min gap between sends (ms):
              <input
                type="number"
                min="1000"
                step="500"
                value={watchMinGapMs}
                onChange={e => setWatchMinGapMs(parseInt(e.target.value || '0', 10))}
                style={{ width: 140, marginLeft: 8 }}
              />
            </label>
          </div>

          <div className="send-controls">
            <button type="submit">Send to AI</button>
            <button
              type="button"
              className={statusClass(recording)}
              onClick={recording ? stopRecording : startRecording}
            >
              {recording ? 'Stop Recording' : 'Record Mic'}
            </button>
          </div>
        </form>
      </div>
      <div className="right-panel">
        <h1>Pending AI Reply</h1>
        {pending ? (
          <form onSubmit={sendPendingToDiscord}>
            <textarea
              name="reply"
              rows="4"
              cols="50"
              value={pending}
              onChange={e => setPending(e.target.value)}
            ></textarea><br />
            <button type="submit" disabled={!discordEnabled}>Send to Discord</button>
            <button type="button" onClick={clearPending}>Clear</button>
          </form>
        ) : (
          <p>No pending message.</p>
        )}
        <button
          className={statusClass(discordEnabled)}
          onClick={() => updateDiscordEnabled(!discordEnabled)}
        >
          Send to Discord: {discordEnabled ? 'On' : 'Off'}
        </button>
        <button
          className={statusClass(ttsEnabled)}
          onClick={toggleTts}
        >
          TTS: {ttsEnabled ? 'On' : 'Off'}
        </button>
        <div className="timings">
          <h2>Task Timings (ms)</h2>
          <p>Speech Detection: {timings.speech_ms}</p>
          <p>LLM: {timings.llm_ms}</p>
          <p>TTS: {timings.tts_ms}</p>
          <p>Total: {timings.total_ms}</p>
        </div>
        <h2>Discord Bot</h2>
        <button
          className={statusClass(botRunning)}
          onClick={toggleBot}
        >
          {botRunning ? 'Stop Bot' : 'Start Bot'}
        </button>
        <h2>Speech Recognition</h2>
        <button
          className={statusClass(speechEnabled)}
          onClick={() => updateSpeechEnabled(!speechEnabled)}
        >
          Speech Recognition: {speechEnabled ? 'On' : 'Off'}
        </button>
        <h2>Voice Control</h2>
        <form method="post" action="voice">
          <input type="hidden" name="action" value="join" />
          <input type="text" name="channel_id" placeholder="Voice Channel ID" /><br />
          <button type="submit">Join Voice</button>
        </form>
        <form method="post" action="voice">
          <input type="hidden" name="action" value="join" />
          <input type="hidden" name="channel_id" value="1402626902694432798" />
          <button type="submit">Join Bot Channel</button>
        </form>
        <form method="post" action="voice">
          <input type="hidden" name="action" value="leave" />
          <button type="submit">Leave Voice</button>
        </form>
        <h2>Piper Voice Settings</h2>
        <div className="piper-settings">
          <label>
            Voice:
            <input
              type="text"
              value={piperSettings.voice}
              onChange={e => updatePiperSetting('voice', e.target.value)}
            />
          </label><br />
          <label>
            Rate:
            <input
              type="text"
              value={piperSettings.rate}
              onChange={e => updatePiperSetting('rate', e.target.value)}
            />
          </label><br />
          <label>
            Pitch ST:
            <input
              type="text"
              value={piperSettings.pitch_st}
              onChange={e => updatePiperSetting('pitch_st', e.target.value)}
            />
          </label><br />
          <label>
            Atempo:
            <input
              type="text"
              value={piperSettings.atempo}
              onChange={e => updatePiperSetting('atempo', e.target.value)}
            />
          </label><br />
          <label>
            Length Scale:
            <input
              type="text"
              value={piperSettings.length_scale}
              onChange={e => updatePiperSetting('length_scale', e.target.value)}
            />
          </label><br />
          <label>
            Noise Scale:
            <input
              type="text"
              value={piperSettings.noise_scale}
              onChange={e => updatePiperSetting('noise_scale', e.target.value)}
            />
          </label><br />
          <label>
            Noise W:
            <input
              type="text"
              value={piperSettings.noise_w}
              onChange={e => updatePiperSetting('noise_w', e.target.value)}
            />
          </label><br />
          <button type="button" onClick={savePiperSettings}>Save</button>
        </div>
        <h2>Log</h2>
        <button onClick={() => setShowLog(!showLog)}>
          {showLog ? 'Hide Log' : 'Show Log'}
        </button>
        {showLog && <pre className="log-window">{log}</pre>}
      </div>
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root'));
