function App() {
  const [conversation, setConversation] = React.useState([]);
  const [pending, setPending] = React.useState('');
  const [recording, setRecording] = React.useState(false);
  const [textChannelId, setTextChannelId] = React.useState('');
  const [userName, setUserName] = React.useState('');
  const [userText, setUserText] = React.useState('');
  const [speechEnabled, setSpeechEnabled] = React.useState(true);
  const [discordEnabled, setDiscordEnabled] = React.useState(true);
  const [ttsEnabled, setTtsEnabled] = React.useState(true);
  const [log, setLog] = React.useState('');
  const [showLog, setShowLog] = React.useState(false);
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
        .then(data => setDiscordEnabled(data.enabled));
      fetch('/log')
        .then(res => res.json())
        .then(data => setLog(data.log));
      fetch('/piper_settings')
        .then(res => res.json())
        .then(data => setPiperSettings(data));
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
      await fetch('/queue_audio', { method: 'POST', body: formData });
    };
    mediaRecorderRef.current.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  const sendText = async (e) => {
    e.preventDefault();
    const payload = {
      user_message: userText,
      user_name: userName,
    };
    if (textChannelId) payload.channel_id = textChannelId;
    await fetch('/queue', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    setUserText('');
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

  const startBot = async () => {
    await fetch('/discord_bot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'start' }),
    });
  };

  const stopBot = async () => {
    await fetch('/discord_bot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'stop' }),
    });
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

  return (
    <div className="container">
      <div className="left-panel">
        <h2>Conversation</h2>
        <div className="conversation">
          {conversation.map((c, i) => (
            <p key={i}>
              <b>{c.user_name}:</b> {c.user_message}
              <br />
              <b>Anna Bortion:</b> {c.reply}
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
          <div className="send-controls">
            <button type="submit">Send to AI</button>
            <button type="button" onClick={recording ? stopRecording : startRecording}>
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
        <button onClick={() => updateDiscordEnabled(!discordEnabled)}>
          Send to Discord: {discordEnabled ? 'On' : 'Off'}
        </button>
        <button onClick={() => setTtsEnabled(!ttsEnabled)}>
          TTS: {ttsEnabled ? 'On' : 'Off'}
        </button>
        <h2>Discord Bot</h2>
        <button onClick={startBot}>Start Bot</button>
        <button onClick={stopBot}>Stop Bot</button>
        <h2>Speech Recognition</h2>
        <p>Status: {speechEnabled ? 'Enabled' : 'Disabled'}</p>
        <button onClick={() => updateSpeechEnabled(false)}>Disable Speech Recognition</button>
        <button onClick={() => updateSpeechEnabled(true)}>Enable Speech Recognition</button>
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
