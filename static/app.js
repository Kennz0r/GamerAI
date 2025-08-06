function App() {
  const [conversation, setConversation] = React.useState([]);
  const [pending, setPending] = React.useState('');
  const [recording, setRecording] = React.useState(false);
  const [textChannelId, setTextChannelId] = React.useState('');
  const [userName, setUserName] = React.useState('');
  const [userText, setUserText] = React.useState('');
  const [aiHearing, setAiHearing] = React.useState(true);
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
      fetch('/ai_enabled')
        .then(res => res.json())
        .then(data => setAiHearing(data.enabled));
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

  const updateAiHearing = async enabled => {
    await fetch('/ai_enabled', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    setAiHearing(enabled);
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
      <h2>AI Hearing</h2>
      <p>Status: {aiHearing ? 'Enabled' : 'Disabled'}</p>
      <button onClick={() => updateAiHearing(false)}>Disable Hearing</button>
      <button onClick={() => updateAiHearing(true)}>Enable Hearing</button>
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
      {conversation.map((c, i) => (
        <p key={i}>
          <b>{c.user_name}:</b> {c.user_message}
          <br />
          <b>AI:</b> {c.reply}
        </p>
      ))}
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root'));
