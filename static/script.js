const form = document.getElementById('chat-form');
const input = document.getElementById('question-input');
const chatHistory = document.getElementById('chat-history');
const sendBtn = document.getElementById('send-btn');

function appendMessage(role, text, meta = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    
    let innerHTML = `<div class="message-content">${text.replace(/\n/g, '<br>')}</div>`;
    
    if (meta) {
        let metaHtml = `<div class="meta-info">`;
        metaHtml += `<span>Status: ${meta.classification}</span>`;
        if (meta.confidence) metaHtml += `<span>Conf: ${(meta.confidence * 100).toFixed(0)}%</span>`;
        metaHtml += `</div>`;
        
        if (meta.sources && meta.sources.length > 0) {
            metaHtml += `<div class="sources"><strong>Sources:</strong><ul>`;
            const uniqueSources = [...new Set(meta.sources.map(s => s.source_id))];
            uniqueSources.forEach(s => {
                metaHtml += `<li>${s}</li>`;
            });
            metaHtml += `</ul></div>`;
        }
        
        innerHTML = `<div class="message-content">${text.replace(/\n/g, '<br>')}${metaHtml}</div>`;
    }
    
    msgDiv.innerHTML = innerHTML;
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function showTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message system-message typing-container';
    div.id = 'typing-indicator';
    div.innerHTML = `
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
    `;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    appendMessage('user', question);
    input.value = '';
    sendBtn.disabled = true;
    showTypingIndicator();

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        removeTypingIndicator();
        
        let answerText = data.answer;
        if (data.clarification_question) {
            answerText += "\n\n" + data.clarification_question;
        }

        appendMessage('system', answerText, data);
    } catch (error) {
        console.error(error);
        removeTypingIndicator();
        appendMessage('system', 'Sorry, there was an error connecting to the agent API.');
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
});
