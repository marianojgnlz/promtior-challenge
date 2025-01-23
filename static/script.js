document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    function createMessageElement(isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `max-w-[80%] ${isUser ? 'ml-auto' : 'mr-auto'} animate-fade-in relative group flex items-center gap-2`;
        
        const messageBubble = document.createElement('div');
        messageBubble.className = `rounded-2xl px-4 py-2 ${
            isUser 
                ? 'bg-primary text-white rounded-br-sm' 
                : 'bg-light text-gray-800 rounded-bl-sm'
        }`;
        
        const copyButton = document.createElement('button');
        copyButton.className = 'flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity p-2 rounded-full hover:bg-gray-200 bg-gray-100';
        copyButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
            </svg>
        `;
        copyButton.title = 'Copy message';
        
        copyButton.onclick = (e) => {
            e.stopPropagation();
            navigator.clipboard.writeText(messageBubble.textContent).then(() => {
                const originalSvg = copyButton.innerHTML;
                copyButton.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                    </svg>
                `;
                setTimeout(() => {
                    copyButton.innerHTML = originalSvg;
                }, 2000);
            });
        };
        
        messageDiv.appendChild(isUser ? copyButton : messageBubble);
        messageDiv.appendChild(isUser ? messageBubble : copyButton);
        return { messageDiv, messageBubble };
    }

    function addMessage(content, isUser) {
        const { messageDiv, messageBubble } = createMessageElement(isUser);
        messageBubble.textContent = content;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function sendMessage() {
        try {
            const message = userInput.value.trim();
            if (!message) return;

            addMessage(message, true);
            userInput.value = '';
            sendButton.disabled = true;

            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Request failed');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let aiMessage = '';
            let messageDiv = document.createElement('div');
            messageDiv.className = 'message ai-message';
            chatMessages.appendChild(messageDiv);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const data = JSON.parse(chunk.replace('data: ', ''));
                
                if (data.error) {
                    messageDiv.textContent = `Error: ${data.error}`;
                    break;
                }
                if (data.content) {
                    aiMessage += data.content;
                    messageDiv.textContent = aiMessage;
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
            }
        } catch (error) {
            console.error('Chat error:', error);
            const errorDiv = document.createElement('div');
            errorDiv.className = 'message error-message';
            errorDiv.textContent = `Error: ${error.message}`;
            chatMessages.appendChild(errorDiv);
        } finally {
            sendButton.disabled = false;
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
}); 