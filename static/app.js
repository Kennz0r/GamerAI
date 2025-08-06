function App() {
  const [conversation, setConversation] = React.useState([]);
  const [pending, setPending] = React.useState('');
  const [recording, setRecording] = React.useState(false);
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
      <h2>Conversation</h2>
      {conversation.map((c, i) => (
        <p key={i}>
          <b>User:</b> {c.user}
          <br />
          <b>AI:</b> {c.reply}
        </p>
      ))}
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root'));
