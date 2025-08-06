function App() {
  const [conversation, setConversation] = React.useState([]);
  const [pending, setPending] = React.useState('');

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
