const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const fileUpload = document.getElementById("fileUpload");
const uploadBtn = document.getElementById("uploadBtn");
const clearBtn = document.getElementById("clearBtn");
const uploadStatus = document.getElementById("uploadStatus");

// Helper function to format timestamp
function getTimestamp() {
    const now = new Date();
    return now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

// Helper function to create message element
function createMessage(content, type) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${type}-message`;
    
    const contentDiv = document.createElement("div");
    if (type === "bot") {
        // Render markdown for bot messages
        contentDiv.innerHTML = marked.parse(content);
    } else {
        contentDiv.textContent = content;
    }
    
    const timestamp = document.createElement("div");
    timestamp.className = "timestamp";
    timestamp.textContent = getTimestamp();
    
    msgDiv.appendChild(contentDiv);
    msgDiv.appendChild(timestamp);
    
    return msgDiv;
}

// Helper function to create typing indicator
function createTypingIndicator() {
    const typingDiv = document.createElement("div");
    typingDiv.className = "typing-indicator";
    typingDiv.id = "typing";
    typingDiv.innerHTML = "<span></span><span></span><span></span>";
    return typingDiv;
}

// Send message function
const sendMsg = async function() {
    const text = input.value.trim();
    if (!text) return;

    // Disable input while processing
    sendBtn.disabled = true;
    input.disabled = true;

    // Display user message
    const userMsg = createMessage(text, "user");
    chat.appendChild(userMsg);
    chat.scrollTop = chat.scrollHeight;

    input.value = "";

    // Show typing indicator
    const typingIndicator = createTypingIndicator();
    chat.appendChild(typingIndicator);
    chat.scrollTop = chat.scrollHeight;

    try {
        // Make the API call to the FastAPI backend
        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: text })
        });
        
        // Remove typing indicator
        typingIndicator.remove();
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const botMessage = data.response;

        // Display the bot's response with markdown rendering
        const botMsg = createMessage(botMessage, "bot");
        chat.appendChild(botMsg);
        chat.scrollTop = chat.scrollHeight;

    } catch (error) {
        // Remove typing indicator
        typingIndicator.remove();
        
        console.error("Error fetching bot response:", error);
        const errorMsg = createMessage("Could not connect to the server. Make sure the backend is running.", "error");
        chat.appendChild(errorMsg);
        chat.scrollTop = chat.scrollHeight;
    } finally {
        // Re-enable input
        sendBtn.disabled = false;
        input.disabled = false;
        input.focus();
    }
};

// Upload PDF function
const uploadPDF = async function() {
    const file = fileUpload.files[0];
    if (!file) {
        uploadStatus.textContent = "Please select a file first";
        uploadStatus.style.color = "#ff6b6b";
        return;
    }

    if (!file.name.endsWith('.pdf')) {
        uploadStatus.textContent = "Only PDF files are allowed";
        uploadStatus.style.color = "#ff6b6b";
        return;
    }

    uploadBtn.disabled = true;
    uploadStatus.textContent = "Uploading...";
    uploadStatus.style.color = "#ffa500";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/upload", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        uploadStatus.textContent = "✓ " + data.message;
        uploadStatus.style.color = "#4CAF50";
        fileUpload.value = "";
        
        // Clear message after 5 seconds
        setTimeout(() => {
            uploadStatus.textContent = "";
        }, 5000);

    } catch (error) {
        console.error("Error uploading file:", error);
        uploadStatus.textContent = "✗ Upload failed";
        uploadStatus.style.color = "#ff6b6b";
        
        // Clear message after 5 seconds
        setTimeout(() => {
            uploadStatus.textContent = "";
        }, 5000);
    } finally {
        uploadBtn.disabled = false;
    }
};

// Clear documents function
const clearDocuments = async function() {
    if (!confirm("Are you sure you want to clear all documents? This cannot be undone.")) {
        return;
    }

    clearBtn.disabled = true;
    uploadStatus.textContent = "Clearing...";
    uploadStatus.style.color = "#ffa500";

    try {
        const response = await fetch("http://127.0.0.1:8000/clear", {
            method: "POST"
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        uploadStatus.textContent = "✓ " + data.message;
        uploadStatus.style.color = "#4CAF50";
        
        setTimeout(() => {
            uploadStatus.textContent = "";
        }, 5000);

    } catch (error) {
        console.error("Error clearing documents:", error);
        uploadStatus.textContent = "✗ Failed to clear documents";
        uploadStatus.style.color = "#ff6b6b";
        
        setTimeout(() => {
            uploadStatus.textContent = "";
        }, 5000);
    } finally {
        clearBtn.disabled = false;
    }
};

// Event listeners
sendBtn.addEventListener("click", sendMsg);

input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !sendBtn.disabled) {
        event.preventDefault();
        sendMsg();
    }
});

uploadBtn.addEventListener("click", uploadPDF);
clearBtn.addEventListener("click", clearDocuments);