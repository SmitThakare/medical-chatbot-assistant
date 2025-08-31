import chainlit as cl
from model import qa_bot  # make sure model.py exists in the same folder

@cl.on_chat_start
async def start():
    chain = qa_bot()
    cl.user_session.set("chain", chain)
    await cl.Message(content="👋 Hi! I’m your Medical Bot. Ask me any medical question.").send()

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="❌ QA chain not initialized.").send()
        return
    
    res = chain.invoke({"query": message.content})
    answer = res.get("result", "No answer found")
    await cl.Message(content=answer).send()