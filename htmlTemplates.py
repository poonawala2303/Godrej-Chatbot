css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    align-items: center;
}

.chat-message.user {
    background-color: #2b313e;
    flex-direction: row-reverse;
}

.chat-message.bot {
    background-color: #475063;
}

.chat-message .avatar {
    flex: 0 0 70px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 1rem;
}

.chat-message.user .avatar {
    margin-left: 1rem;
    margin-right: 0;
}

.chat-message .avatar img {
    max-width: 70px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    flex: 1;
    color: #fff;
    padding: 0 1.5rem;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRPYtqytSUUShobf3PDxLMbcfTYj9DmcY3P1Q&s" alt="Bot Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://static.vecteezy.com/system/resources/thumbnails/008/442/086/small/illustration-of-human-icon-user-symbol-icon-modern-design-on-blank-background-free-vector.jpg" alt="User Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
