const form = document.getElementById('example-form');
const statusEl = document.getElementById('train-status');

form.addEventListener('submit', async e => {
  e.preventDefault();
  const prompt = document.getElementById('prompt').value;
  const response = document.getElementById('response').value;
  const res = await fetch('/training_data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, response })
  });
  const data = await res.json();
  statusEl.textContent = JSON.stringify(data, null, 2);
  form.reset();
});

const trainBtn = document.getElementById('train-btn');
trainBtn.addEventListener('click', async () => {
  const res = await fetch('/fine_tune', { method: 'POST' });
  const data = await res.json();
  statusEl.textContent = JSON.stringify(data, null, 2);
});
