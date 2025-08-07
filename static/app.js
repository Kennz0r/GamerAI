function App() {
  const [conversation, setConversation] = React.useState([]);
  const [pending, setPending] = React.useState('');
  const [recording, setRecording] = React.useState(false);
  const [textChannelId, setTextChannelId] = React.useState('1285333390643695769');
  const [userName, setUserName] = React.useState('');
  const [userText, setUserText] = React.useState('');
  const [speechEnabled, setSpeechEnabled] = React.useState(true);
  const [log, setLog] = React.useState('');
  const [showLog, setShowLog] = React.useState(false);
  const mediaRecorderRef = React.useRef(null);
  const chunksRef = React.useRef([]);

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
      fetch('/log')
        .then(res => res.json())
        .then(data => setLog(data.log));
    };
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);
    mediaRecorderRef.current.ondataavailable = e => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    mediaRecorderRef.current.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
      chunksRef.current = [];
      const formData = new FormData();
      formData.append('file', blob, 'audio.webm');
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
    await fetch('/queue', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        channel_id: textChannelId,
        user_message: userText,
        user_name: userName,
      }),
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

  const playTTS = async () => {
    const textarea = document.querySelector('textarea[name="reply"]');
    const text = textarea ? textarea.value : pending;
    const res = await fetch('/tts_preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (res.status === 200) {
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play();
    }
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

  return (
    <div>
      <h1>Pending AI Reply</h1>
      {pending ? (
        <form method="post" action="approve">
          <textarea name="reply" rows="4" cols="50" defaultValue={pending}></textarea><br />
          <button type="submit">Send to Discord</button>
        </form>
      ) : (
        <p>No pending message.</p>
      )}
      <button onClick={playTTS}>Play TTS</button>
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
      <h2>Microphone</h2>
      <button onClick={recording ? stopRecording : startRecording}>
        {recording ? 'Stop Recording' : 'Record Mic'}
      </button>
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
        ></textarea><br />
        <button type="submit">Send to AI</button>
      </form>
      <h2>Conversation</h2>
      <div className="conversation">
        {conversation.map((c, i) => (
          <p key={i}>
            <b>{c.user_name}:</b> {c.user_message}
            <br />
            <b>AI:</b> {c.reply}
          </p>
        ))}
      </div>
      <h2>Log</h2>
      <button onClick={() => setShowLog(!showLog)}>
        {showLog ? 'Hide Log' : 'Show Log'}
      </button>
      {showLog && <pre className="log-window">{log}</pre>}
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root'));
