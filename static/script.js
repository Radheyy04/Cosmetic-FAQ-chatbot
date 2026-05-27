async function sendMessage() {

    const input = document.getElementById("userInput");
    const message = input.value;

    if (message === "") return;

    const chatBox = document.getElementById("chatBox");

    // User Message
    chatBox.innerHTML += `
        <div class="message user">${message}</div>`;

    // Send to Backend
    const response = await fetch("/chat", {

        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            message: message
        })
    });

    const data = await response.json();

    // Bot Reply
    chatBox.innerHTML += `
        <div class="message bot">${data.reply}</div>
    `;
    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;
}
