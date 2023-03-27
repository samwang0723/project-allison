let activeDiv = null;
let currentMsg = "";
let refreshBottom = true;

$(document).ready(function() {
    var socket = io.connect("http://" + document.domain + ":" + location.port, {
        transports: ["websocket"],
    });

    socket.on("connect", function() {
        console.log("Connected to the server.");
    });

    socket.on("message", function(data) {
        // If the message is "[[stop]]", reset the activeDiv
        if (data === "[[stop]]") {
            activeDiv = null;
            currentMsg = "";
            return;
        }

        currentMsg += data;
        if (!activeDiv) {
            addMessageRow("allison");
        }
        formatMessage(currentMsg);
    });

    $("#chat_form").on("submit", function(e) {
        e.preventDefault();
        var message = $("#message-textfield").val();
        if (message === "") {
            return;
        }

        addMessageRow("user");
        formatMessage(message);

        socket.emit("message", message);

        // reset the input field and cache values
        $("#message-textfield").val("");
        $("#message-textfield").height(40);
        activeDiv = null;
        currentMsg = "";
    });

    const messageInput = document.getElementById("message-textfield");
    messageInput.addEventListener("keydown", function(event) {
        if (event.key === "Enter" && event.shiftKey) {
            event.preventDefault();
            const value = this.value;
            this.value = value + "\n";
        }
    });
    messageInput.oninput = function() {
        messageInput.style.height = "52px";
        messageInput.style.height = Math.min(messageInput.scrollHeight, 400) + "px";
    };

    const messages = document.getElementById("messages");
    messages.addEventListener("scroll", function() {
        // Check if the user just scrolled
        if (
            messages.scrollTop + messages.clientHeight >=
            messages.scrollHeight - 60
        ) {
            // User scrolled to the bottom, do something
            refreshBottom = true;
        } else {
            // User scrolled, but not to the bottom, do something else
            refreshBottom = false;
        }
    });
});

function linkify(inputText) {
    var replacedText, replacePattern1, replacePattern2, replacePattern3;

    //URLs starting with http://, https://, or ftp://
    replacePattern1 =
        /(\b(https?|ftp):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/gim;
    replacedText = inputText.replace(replacePattern1, "[$1]($1)");

    //URLs starting with "www." (without // before it, or it'd re-link the ones done above).
    replacePattern2 = /(^|[^\/])(www\.[\S]+(\b|$))/gim;
    replacedText = replacedText.replace(replacePattern2, "[$1]($2)");

    //Change email addresses to mailto:: links.
    replacePattern3 = /(([a-zA-Z0-9\-\_\.])+@[a-zA-Z\_]+?(\.[a-zA-Z]{2,6})+)/gim;
    replacedText = replacedText.replace(replacePattern3, "[$1](mailto:$1)");

    return replacedText;
}

function boldify(inputText) {
    var replacedText, replacePattern1;

    replacePattern1 =
        /(Subject:|Summary:|Description:|Sources:|Attachments:|Similarity:|Prompt:)/gim;
    replacedText = inputText.replace(replacePattern1, "___$1___");

    return replacedText;
}

function addMessageRow(sender) {
    let messageRow = document.createElement("div");
    messageRow.classList.add("message-row");

    let messageSender = document.createElement("span");
    messageSender.classList.add("message-sender");
    messageSender.innerHTML =
        '<img width="50px" height="50px" src="static/' + sender + '.svg">';
    messageRow.appendChild(messageSender);

    let messageText = document.createElement("span");
    messageText.classList.add("message-body");
    activeDiv = messageText;
    messageRow.appendChild(messageText);

    let messageTail = document.createElement("span");
    messageTail.classList.add("message-tail");
    messageRow.appendChild(messageTail);

    let messages = document.getElementById("messages");
    messages.appendChild(messageRow);
}

function formatMessage(message) {
    const lines = message.split("```");
    let output = "";

    for (let i = 0; i < lines.length; i++) {
        msg = lines[i];
        if (i % 2 === 1) {
            let code_lines = msg.split("\n");
            let language = code_lines.shift().trim(); // Remove the first line, which contains the language identifier.
            let code = code_lines.join("\n");
            if (language === "" || language === "html") {
                code = code.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            }

            code_class = "language-";
            if (language != "") {
                code_class = "language-" + language;
            }
            output +=
                '<pre class="prettyprint line-numbers language-markup">' +
                '<code class="' +
                code_class +
                '">' +
                code +
                "</code>" +
                "</pre>";
        } else {
            linkified = linkify(msg);
            boldified = boldify(linkified);
            const md = window.markdownit();
            const outputText = md.render(boldified);
            output += outputText;
        }
    }

    console.log(output);
    activeDiv.innerHTML = output;
    // Apply Prism.js syntax highlighting to the newly added code block(s).
    Prism.highlightAllUnder(activeDiv);

    if (refreshBottom) {
        let messages = document.getElementById("messages");
        messages.scrollTop = messages.scrollHeight;
    }
}