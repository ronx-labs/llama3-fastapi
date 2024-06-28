import os
import torch.distributed as dist
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from llama import Dialog, Llama

app = FastAPI()

# llm generator
generator = None

class Message(BaseModel):
    role: str
    content: str
    
class InputData(BaseModel):
    messages_batch: List[List[Message]]


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI server!"}

@app.post("/inference/")
def do_inference(data: InputData):
    global generator

    max_gen_len = 512
    temperature = 0
    top_p = 0.9

    messages_batch = data.messages_batch
    dialogs: List[Dialog] = [
        [
            {
                "role": message.role,
                "content": message.content
            } for message in messages
        ] for messages in messages_batch
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    return results

@app.on_event("startup")
async def startup_event():
    print("===> Initializing distributed process group")
    
    # Set environment variables for torch distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = os.environ.get('RANK', '0')
    os.environ['WORLD_SIZE'] = os.environ.get('WORLD_SIZE', '1')
    
    dist.init_process_group(backend='nccl', init_method='env://')
    
    print("===> Loading Llama3-8b-Instruct model")
    
    global generator
    
    ckpt_dir = "Meta-Llama-3-8B-Instruct"
    tokenizer_path = "Meta-Llama-3-8B-Instruct/tokenizer.model"
    max_seq_len = 8192
    max_batch_size = 20
    
    # Initialize the global variable
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    print("===> Successfully loaded the llm model")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
